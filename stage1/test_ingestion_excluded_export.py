"""The excluded export keeps provenance while using an equijoin."""

import unittest

import duckdb

from stage1.ingestion.sql import AUXILIARY_EXPORTS


LEGACY_QUERY = """SELECT r.*, f.source AS duplicate_of_source,
    f.source_row AS duplicate_of_source_row FROM classified r
    LEFT JOIN raw f ON r.first_row_id=f.row_id AND r.disposition='exact_duplicate'
    WHERE r.disposition<>'accepted' ORDER BY r.row_id"""


class ExcludedExportTests(unittest.TestCase):
    def test_export_matches_original_for_all_exclusion_dispositions(self):
        connection = duckdb.connect()
        self.addCleanup(connection.close)
        connection.execute("""CREATE TABLE raw (
            row_id BIGINT, source VARCHAR, source_row BIGINT
        )""")
        connection.execute("""INSERT INTO raw VALUES
            (1, 'a.csv', 2), (2, 'a.csv', 3), (3, 'a.csv', 4),
            (4, 'b.csv', 2), (5, 'b.csv', 3), (6, 'b.csv', 4)
        """)
        connection.execute("""CREATE TABLE classified (
            row_id BIGINT, source VARCHAR, source_row BIGINT,
            first_row_id BIGINT, disposition VARCHAR, value_raw VARCHAR
        )""")
        connection.execute("""INSERT INTO classified VALUES
            (1, 'a.csv', 2, 1, 'accepted', 'x'),
            (2, 'a.csv', 3, 1, 'exact_duplicate', 'x'),
            (3, 'a.csv', 4, 3, 'repeated_header', 'header'),
            (4, 'b.csv', 2, 4, 'quarantine', 'bad'),
            (5, 'b.csv', 3, 1, 'exact_duplicate', 'x'),
            (6, 'b.csv', 4, 4, 'exact_duplicate', 'bad')
        """)

        original = connection.execute(LEGACY_QUERY).to_arrow_table()
        optimized = connection.execute(AUXILIARY_EXPORTS["excluded_rows"]).to_arrow_table()
        self.assertTrue(optimized.equals(original, check_metadata=True))
        self.assertEqual(optimized.column("row_id").to_pylist(), [2, 3, 4, 5, 6])
        self.assertEqual(
            optimized.column("duplicate_of_source").to_pylist(),
            ["a.csv", None, None, "a.csv", "b.csv"],
        )

        plan = connection.execute("EXPLAIN " + AUXILIARY_EXPORTS["excluded_rows"]).fetchone()[1]
        self.assertIn("HASH_JOIN", plan)
        self.assertNotIn("BLOCKWISE_NL_JOIN", plan)


if __name__ == "__main__":
    unittest.main()
