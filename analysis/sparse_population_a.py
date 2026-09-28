"""Vectorized causal guards equivalent to SparseAdmissionStream on M1.

No labels, A3 statuses or retrospective R3 keys enter these calculations.
Type-changing channels use the original B state machine, not a SQL shortcut.
"""

from __future__ import annotations

from itertools import groupby
from datetime import datetime, timedelta

import pyarrow as pa

from analysis.build_quality_improvement_a import quoted
from stage1.features.hourly import HourlyConfig
from stage1.features.sparse_admission import VERSION
from stage1.state_labeling.registered_episodes import EpisodeBuilder, StateEvent
from stage1.state_labeling.operational import registered_state_effect
from stage1.state_labeling.rules import KNOWN_SENSOR_TYPES


KNOWN = ",".join(quoted(value) for value in sorted(KNOWN_SENSOR_TYPES))
FLAGS = ",".join(quoted(value) for value in sorted(HourlyConfig().excluded_quality_flags))


def build_past_prefixes(db, files: list[str]) -> dict:
    db.execute(
        "CREATE TEMP VIEW raw_events AS SELECT *,"
        "CASE WHEN year(timestamp)<2021 THEN 0 ELSE 1 END AS archive_segment,"
        f"NOT list_has_any(COALESCE(quality_flags,[]),[{FLAGS}]) AS usable "
        "FROM read_parquet(["
        + ",".join(quoted(path) for path in files)
        + "],hive_partitioning=false) WHERE "
        "timestamp>=TIMESTAMP '2019-01-01' AND timestamp<TIMESTAMP '2026-01-01' "
        "AND year(timestamp)<>2021 AND "
        "split_part(replace(source,chr(92),'/'),'/',-1)="
        "'ext-journal-' || CAST(year(timestamp) AS VARCHAR) || '.7z'",
    )
    db.execute(
        "CREATE TEMP TABLE first_history AS SELECT channel_id,archive_segment,"
        "MIN(timestamp) AS first_observation_at,"
        "MIN(timestamp) FILTER(WHERE usable) AS first_usable_at "
        "FROM raw_events GROUP BY ALL"
    )
    db.execute(
        "CREATE TEMP TABLE history AS SELECT h.*,"
        "MIN(e.timestamp) FILTER(WHERE e.usable AND e.timestamp>h.first_usable_at) "
        "AS second_usable_at FROM first_history h JOIN raw_events e "
        "USING(channel_id,archive_segment) GROUP BY ALL"
    )
    db.execute(
        "CREATE TEMP TABLE first_types AS SELECT channel_id,archive_segment,sensor_type,"
        "MIN(timestamp) AS timestamp FROM raw_events WHERE sensor_type IS NOT NULL "
        "AND sensor_type<>'' GROUP BY ALL"
    )
    db.execute(
        "CREATE TEMP TABLE type_prefix AS SELECT channel_id,archive_segment,timestamp,"
        "MIN(low_type) OVER w AS low_type,MAX(high_type) OVER w AS high_type FROM "
        "(SELECT channel_id,archive_segment,timestamp,MIN(sensor_type) low_type,"
        "MAX(sensor_type) high_type FROM first_types GROUP BY ALL) "
        "WINDOW w AS (PARTITION BY channel_id,archive_segment ORDER BY timestamp)"
    )
    db.execute(
        f"CREATE TEMP TABLE unknown_types AS SELECT DISTINCT channel_id,archive_segment,"
        f"timestamp FROM raw_events WHERE sensor_type IS NULL OR sensor_type NOT IN ({KNOWN})"
    )
    pairs = db.execute(
        "SELECT DISTINCT sensor_type,value_state FROM raw_events WHERE value_state IS NOT NULL"
    ).fetchall()
    meaning = pa.Table.from_pylist(
        [
            {
                "sensor_type": kind,
                "value_state": text,
                "effect": registered_state_effect(kind, text, False),
            }
            for kind, text in pairs
        ],
        schema=pa.schema(
            [("sensor_type", pa.string()), ("value_state", pa.string()), ("effect", pa.string())]
        ),
    )
    db.register("meaning", meaning)
    db.execute(
        "CREATE TEMP TABLE text_groups(channel_id VARCHAR,archive_segment INTEGER,"
        "timestamp TIMESTAMP,conflict BOOLEAN,fault BOOLEAN,normal BOOLEAN,uncertain BOOLEAN)"
    )
    db.execute(
        "CREATE TEMP TABLE ambiguities(channel_id VARCHAR,archive_segment INTEGER,"
        "timestamp TIMESTAMP)"
    )
    months = db.execute(
        "SELECT DISTINCT strftime(timestamp,'%Y-%m') FROM raw_events "
        "WHERE value_state IS NOT NULL ORDER BY 1"
    ).fetchall()
    for (month,) in months:
        start = datetime.strptime(month, "%Y-%m")
        end = datetime(start.year + (start.month == 12), start.month % 12 + 1, 1)
        db.execute(
            "CREATE OR REPLACE TEMP TABLE text_month AS SELECT e.channel_id,e.archive_segment,e.timestamp,"
            "MIN(e.value_state)<>MAX(e.value_state) OR "
            "MIN(COALESCE(e.sensor_type,'<null>'))<>MAX(COALESCE(e.sensor_type,'<null>')) AS conflict,"
            "BOOL_OR(m.effect='fault') AS fault,BOOL_AND(m.effect='normal') AS normal,"
            "BOOL_OR(m.effect='uncertain') AS uncertain,"
            "COALESCE(MIN(CASE WHEN e.usable THEN e.value_state END)<>"
            "MAX(CASE WHEN e.usable THEN e.value_state END),false) AS usable_ambiguity "
            "FROM raw_events e JOIN meaning m ON e.value_state=m.value_state "
            "AND e.sensor_type IS NOT DISTINCT FROM m.sensor_type "
            "WHERE e.value_state IS NOT NULL AND e.timestamp>=? AND e.timestamp<? GROUP BY ALL",
            [start, end],
        )
        db.execute(
            "INSERT INTO ambiguities SELECT channel_id,archive_segment,timestamp "
            "FROM text_month WHERE usable_ambiguity"
        )
        # Repeated faults/uncertainty do not change admission state. Every
        # normal is retained, since its precise time refreshes the 168h clock.
        db.execute(
            "INSERT INTO text_groups WITH coded AS (SELECT *,"
            "CASE WHEN normal AND NOT conflict THEN 1 ELSE CAST(fault AS INTEGER)*2+"
            "CAST(uncertain OR conflict AS INTEGER)*4 END AS code FROM text_month),"
            "runs AS (SELECT *,LAG(code) OVER(PARTITION BY channel_id,archive_segment ORDER BY timestamp) "
            "AS prior_code FROM coded) SELECT channel_id,archive_segment,timestamp,conflict,fault,normal,"
            "uncertain FROM runs WHERE code<>0 AND (code=1 OR prior_code IS DISTINCT FROM code)"
        )
        print(f"state groups compressed: {month}", flush=True)
    db.execute("DROP TABLE IF EXISTS text_month")
    db.execute(
        "CREATE TEMP TABLE state_prefix AS WITH marks AS (SELECT t.*,"
        "MAX(CASE WHEN normal AND NOT conflict THEN t.timestamp END) OVER w AS normal_at,"
        "MAX(CASE WHEN fault THEN t.timestamp END) OVER w AS fault_at,"
        "MAX(CASE WHEN uncertain OR conflict THEN t.timestamp END) OVER w AS uncertain_at,"
        "MAX(CASE WHEN normal AND NOT conflict AND u.timestamp IS NULL "
        "THEN t.timestamp END) OVER w AS clear_metadata_at "
        "FROM text_groups t LEFT JOIN unknown_types u "
        "USING(channel_id,archive_segment,timestamp) "
        "WINDOW w AS (PARTITION BY t.channel_id,t.archive_segment ORDER BY t.timestamp)) "
        "SELECT channel_id,archive_segment,timestamp,"
        "CASE WHEN (fault_at IS NULL OR normal_at>fault_at) AND "
        "(uncertain_at IS NULL OR normal_at>uncertain_at) THEN normal_at END "
        "AS last_explicit_normal_at,"
        "COALESCE(fault_at>=normal_at,fault_at IS NOT NULL) AS active,"
        "COALESCE(uncertain_at>=normal_at,uncertain_at IS NOT NULL) AS uncertain,"
        "clear_metadata_at FROM marks"
    )
    # Metadata type changes reset B2's live state. Only affected channels need
    # the original Python state machine; dispatch does not select model hours.
    changed = db.execute(
        "SELECT channel_id,archive_segment FROM first_types GROUP BY ALL HAVING COUNT(*)>1"
    ).fetchall()
    if changed:
        _replace_type_changing_states(db, changed)
    for table, condition, count in (
        ("quality_prefix", "NOT usable", "COUNT(*)"),
        ("ambiguity_prefix", "true", "COUNT(*)"),
    ):
        source = "raw_events" if table == "quality_prefix" else "ambiguities"
        db.execute(
            f"CREATE TEMP TABLE {table} AS SELECT channel_id,archive_segment,timestamp,"
            "CAST(SUM(n) OVER (PARTITION BY channel_id,archive_segment ORDER BY timestamp) "
            f"AS BIGINT) AS count FROM (SELECT channel_id,archive_segment,timestamp,{count} n "
            f"FROM {source} WHERE {condition} GROUP BY ALL)"
        )
    return {
        "accepted_events": db.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0],
        "unique_type_text_pairs": len(pairs),
        "type_changing_channel_segments": len(changed),
    }


def retain_full_prefixes(db):
    for name in (
        "state_prefix",
        "unknown_types",
        "quality_prefix",
        "ambiguity_prefix",
        "qa_prefix",
    ):
        db.execute(f"ALTER TABLE {name} RENAME TO {name}_all")


def slice_month_prefixes(db, month):
    """Window data plus one cumulative predecessor preserves all earlier history."""
    start = datetime.strptime(month, "%Y-%m")
    end = datetime(start.year + (start.month == 12), start.month % 12 + 1, 1)
    for name, hours in (
        ("state_prefix", 0),
        ("unknown_types", 0),
        ("quality_prefix", 24),
        ("ambiguity_prefix", 24),
        ("qa_prefix", 168),
    ):
        lower = start - timedelta(hours=hours)
        db.execute(
            f"CREATE OR REPLACE TEMP TABLE {name} AS SELECT * FROM {name}_all "
            "WHERE timestamp>=? AND timestamp<? UNION ALL "
            f"SELECT a.* FROM {name}_all a JOIN (SELECT channel_id,archive_segment,"
            f"MAX(timestamp) AS timestamp FROM {name}_all WHERE timestamp<? GROUP BY ALL) p "
            "USING(channel_id,archive_segment,timestamp)",
            [lower, end, lower],
        )


def _replace_type_changing_states(db, changed):
    selection = pa.Table.from_pylist([{"channel_id": c, "archive_segment": s} for c, s in changed])
    db.register("changed", selection)
    records = db.execute(
        "SELECT e.row_id,e.channel_id,e.archive_segment,e.timestamp,e.sensor_type,"
        "e.value_state,e.alarm, e.sensor_type IS NULL OR "
        f"e.sensor_type NOT IN ({KNOWN}) AS unknown_type "
        "FROM raw_events e SEMI JOIN changed USING(channel_id,archive_segment) "
        "ORDER BY timestamp,channel_id,row_id"
    ).to_arrow_reader(batch_size=25_000)
    builder = EpisodeBuilder()
    clears, result = {}, []
    rows = (row for batch in records for row in batch.to_pylist())
    for (at, channel), group in groupby(rows, lambda row: (row["timestamp"], row["channel_id"])):
        members = list(group)
        texts = [row for row in members if row["value_state"] is not None]
        if not texts:
            continue
        for row in texts:
            builder.add(
                StateEvent(
                    row["row_id"], channel, row["sensor_type"], at, row["value_state"], row["alarm"]
                )
            )
        builder.finish()
        state = builder.states[channel]
        segment = members[0]["archive_segment"]
        if state.last_normal_at == at and not any(row["unknown_type"] for row in members):
            clears[channel, segment] = at
        result.append(
            {
                "channel_id": channel,
                "archive_segment": segment,
                "timestamp": at,
                "last_explicit_normal_at": state.last_normal_at,
                "active": state.open_episode is not None,
                "uncertain": state.uncertain_since_normal,
                "clear_metadata_at": clears.get((channel, segment)),
            }
        )
        builder.episodes[:] = [state.open_episode] if state.open_episode else []
        if state.open_episode:
            state.open_episode.evidence.clear()
    schema = pa.schema(
        [
            ("channel_id", pa.string()),
            ("archive_segment", pa.int32()),
            ("timestamp", pa.timestamp("us")),
            ("last_explicit_normal_at", pa.timestamp("us")),
            ("active", pa.bool_()),
            ("uncertain", pa.bool_()),
            ("clear_metadata_at", pa.timestamp("us")),
        ]
    )
    db.register("replacement_states", pa.Table.from_pylist(result, schema=schema))
    db.execute(
        "DELETE FROM state_prefix USING changed "
        "WHERE state_prefix.channel_id=changed.channel_id AND "
        "state_prefix.archive_segment=changed.archive_segment"
    )
    db.execute("INSERT INTO state_prefix SELECT * FROM replacement_states")


def decision_sql() -> str:
    """keys has only channel_id, prediction_time, archive_segment; no targets."""
    return (
        """WITH evidence AS (
        SELECT k.*, CASE WHEN ty.low_type=ty.high_type THEN ty.low_type END AS sensor_type,
            h.first_observation_at,h.first_usable_at,h.second_usable_at,
            st.last_explicit_normal_at,COALESCE(st.active,false) AS active,
            COALESCE(st.uncertain,false) AS uncertain,
            un.timestamp IS NOT NULL AND (st.clear_metadata_at IS NULL OR
                un.timestamp>st.clear_metadata_at) AS metadata_uncertain,
            COALESCE(qh.count,0)-COALESCE(ql.count,0) AS excluded_quality_count_24h,
            COALESCE(ah.count,0)-COALESCE(al.count,0) AS ambiguous_seconds_24h,
            GREATEST(ty.timestamp,st.timestamp,un.timestamp,qh.timestamp,ah.timestamp,
                CASE WHEN h.first_usable_at<=k.prediction_time THEN h.first_usable_at END,
                CASE WHEN h.second_usable_at<=k.prediction_time THEN h.second_usable_at END)
                AS admission_evidence_through
        FROM keys k LEFT JOIN history h USING(channel_id,archive_segment)
        ASOF LEFT JOIN type_prefix ty ON k.channel_id=ty.channel_id
            AND k.archive_segment=ty.archive_segment AND k.prediction_time>=ty.timestamp
        ASOF LEFT JOIN state_prefix st ON k.channel_id=st.channel_id
            AND k.archive_segment=st.archive_segment AND k.prediction_time>=st.timestamp
        ASOF LEFT JOIN unknown_types un ON k.channel_id=un.channel_id
            AND k.archive_segment=un.archive_segment AND k.prediction_time>=un.timestamp
        ASOF LEFT JOIN quality_prefix qh ON k.channel_id=qh.channel_id
            AND k.archive_segment=qh.archive_segment AND k.prediction_time>=qh.timestamp
        ASOF LEFT JOIN quality_prefix ql ON k.channel_id=ql.channel_id
            AND k.archive_segment=ql.archive_segment
            AND k.prediction_time-INTERVAL '24 hours'>=ql.timestamp
        ASOF LEFT JOIN ambiguity_prefix ah ON k.channel_id=ah.channel_id
            AND k.archive_segment=ah.archive_segment AND k.prediction_time>=ah.timestamp
        ASOF LEFT JOIN ambiguity_prefix al ON k.channel_id=al.channel_id
            AND k.archive_segment=al.archive_segment
            AND k.prediction_time-INTERVAL '24 hours'>=al.timestamp
    ), flags AS (SELECT *,first_observation_at IS NULL OR first_observation_at>prediction_time AS no_history,
        first_usable_at IS NULL OR first_usable_at>prediction_time AS all_excluded
        FROM evidence), reasons AS (SELECT *,
        CASE WHEN no_history THEN ['no_observations_in_current_archive_segment'] ELSE
        list_sort(list_filter([
            CASE WHEN all_excluded THEN 'all_causal_observations_excluded' END,
            CASE WHEN NOT all_excluded AND sensor_type IS NULL THEN 'sensor_type_unknown_or_conflicting' END,
            CASE WHEN NOT all_excluded AND (first_usable_at>prediction_time-INTERVAL '7 days'
                OR second_usable_at IS NULL OR second_usable_at>prediction_time) THEN 'insufficient_history' END,
            CASE WHEN NOT all_excluded AND excluded_quality_count_24h>0 THEN 'quality_exclusions_24h' END,
            CASE WHEN NOT all_excluded AND ambiguous_seconds_24h>0 THEN 'same_time_state_ambiguity' END,
            CASE WHEN sensor_type IS NULL OR sensor_type NOT IN ("""
        + KNOWN
        + """)
                THEN 'unknown_or_conflicting_sensor_type' END,
            CASE WHEN active THEN 'registered_episode_active_at_t' END,
            CASE WHEN uncertain THEN 'uncertain_past_registered_state' END,
            CASE WHEN metadata_uncertain THEN 'unknown_type_observed_since_explicit_normal' END,
            CASE WHEN last_explicit_normal_at IS NULL OR
                last_explicit_normal_at<prediction_time-INTERVAL '168 hours'
                THEN 'no_recent_explicit_normal_at_t' END
        ], x->x IS NOT NULL)) END AS admission_reasons_without_qa FROM flags)
        SELECT channel_id,prediction_time,sensor_type,
            CASE WHEN NOT no_history AND (all_excluded OR active) THEN 'excluded'
                 WHEN len(admission_reasons_without_qa)>0 THEN 'unknown'
                 ELSE 'eligible' END AS admission_status_without_qa,
            admission_reasons_without_qa,last_explicit_normal_at,admission_evidence_through,
            excluded_quality_count_24h,ambiguous_seconds_24h,
            CASE WHEN first_usable_at<=prediction_time THEN first_usable_at END AS first_usable_at,
            CASE WHEN second_usable_at<=prediction_time THEN second_usable_at END AS second_usable_at
        FROM reasons"""
    )


def finalize_sql() -> str:
    severe = " + ".join(
        f"qa_{category}_count_24h"
        for category in (
            "epoch_value_artifact",
            "temperature_service_code_candidate",
            "gas_above_physical_percent",
        )
    )
    return (
        "WITH joined AS (SELECT d.*,q.* EXCLUDE(channel_id,prediction_time),"
        + severe
        + " AS blocking_qa_count_24h FROM decisions_without_qa d JOIN qa_month q "
        "USING(channel_id,prediction_time)) SELECT *,"
        "CASE WHEN admission_status_without_qa='excluded' THEN 'excluded' "
        "WHEN blocking_qa_count_24h>0 THEN 'unknown' ELSE admission_status_without_qa END "
        "AS admission_status,"
        "CASE WHEN blocking_qa_count_24h>0 THEN list_sort(list_append("
        "admission_reasons_without_qa,'qa_unusable_measurement_24h')) "
        "ELSE admission_reasons_without_qa END AS admission_reasons,"
        + quoted(VERSION)
        + " AS candidate_version,'unknown' AS availability_status FROM joined"
    )
