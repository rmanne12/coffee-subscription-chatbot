import unittest
from src.agents.mood_agent import determine_mood_and_urgency

class TestMoodAgent(unittest.TestCase):
    def test_determine_mood_and_urgency(self):
        conversation_history = []
        result = determine_mood_and_urgency(conversation_history)
        self.assertIn("mood", result)
        self.assertIn("urgency", result)

if __name__ == "__main__":
    unittest.main()
