import tempfile
import unittest
from pathlib import Path

from exam_trainer import HERE, load_tickets


class LoadTicketsTests(unittest.TestCase):
    def test_loads_all_sample_tickets(self):
        tickets = load_tickets(HERE / "sample_answers.md")

        self.assertEqual([1, 2, 3], list(tickets))
        self.assertEqual(
            "Переменные и базовые типы данных в Python",
            tickets[1]["вопрос"],
        )

    def test_each_sample_ticket_contains_expected_sections(self):
        tickets = load_tickets(HERE / "sample_answers.md")

        for number, ticket in tickets.items():
            with self.subTest(ticket=number):
                self.assertTrue(ticket["вопрос"])
                self.assertIn("**Тезисно**", ticket["эталон"])
                self.assertIn("**Примеры**", ticket["эталон"])
                self.assertIn("**Текстом**", ticket["эталон"])

    def test_empty_document_has_no_tickets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answers.md"
            path.write_text("# Конспект без билетов\n", encoding="utf-8")

            self.assertEqual({}, load_tickets(path))


if __name__ == "__main__":
    unittest.main()
