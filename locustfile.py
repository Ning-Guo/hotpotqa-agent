"""
locustfile.py — Locust load test for the HotpotQA Multi-Hop Agent API.

Usage:
    # Interactive UI (recommended for first run):
    locust -f locustfile.py --host http://localhost:8000
    # then open http://localhost:8089 in browser

    # Headless (CI / automated):
    locust -f locustfile.py --host http://localhost:8000 \
        --headless -u 5 -r 1 --run-time 60s

Test coverage:
    - Bridge questions  (2-hop reasoning, Wikipedia fallback)
    - Comparison questions (yes/no, entity comparison)
    - Vector DB retrieval vs Wikipedia web search paths
"""

import random
from locust import HttpUser, task, between

# ---------------------------------------------------------------------------
# Test questions — diverse coverage of agent paths
# ---------------------------------------------------------------------------

# Bridge questions: require 2-hop reasoning (find intermediate entity first)
BRIDGE_WIKI = [
    "What is the capital of the country where the Big Ben is located?",
    "What is the nationality of the director of the film Titanic?",
    "In which city is the headquarters of the company that makes iPhone?",
    "What language is spoken in the country where the Eiffel Tower is located?",
    "What is the name of the university attended by the founder of Facebook?",
]

BRIDGE_VECTOR = [
    "What media organization which posts talks is Jesse Dylan a member of?",
    "Which is the largest and most populous city in the region with the national anthem Hoyamal?",
    "What is the occupation of the person who composed the national anthem of Burkina Faso?",
    "In what country was the actress who played Hermione Granger born?",
    "What sport did the father of Kiefer Sutherland play professionally?",
]

# Comparison questions: yes/no, compare two entities
COMPARE_WIKI = [
    "Were Scott Derrickson and Ed Wood of the same nationality?",
    "Are both The Bridge and Grand Canyon documentary films?",
    "Did Albert Einstein and Isaac Newton both win the Nobel Prize?",
    "Are both Python and Java object-oriented programming languages?",
    "Were both Marie Curie and Rosalind Franklin chemists?",
]

COMPARE_VECTOR = [
    "Are both Subak and Caral originated from the same country?",
    "Do both the Nile and the Amazon flow into the Atlantic Ocean?",
    "Are both the Parthenon and the Colosseum located in Southern Europe?",
    "Were both Nikola Tesla and Thomas Edison born in Europe?",
    "Did both Winston Churchill and Franklin Roosevelt serve during World War II?",
]

ALL_QUESTIONS = BRIDGE_WIKI + BRIDGE_VECTOR + COMPARE_WIKI + COMPARE_VECTOR


# ---------------------------------------------------------------------------
# Locust user
# ---------------------------------------------------------------------------

class AgentUser(HttpUser):
    """
    Simulates a single user sending questions to the agent API.
    wait_time: pause between consecutive requests per user (seconds).
    """
    wait_time = between(2, 5)

    @task(2)
    def ask_bridge(self):
        """Bridge questions — heavier weight, tests multi-hop reasoning."""
        question = random.choice(BRIDGE_WIKI + BRIDGE_VECTOR)
        self._ask(question, label="bridge")

    @task(2)
    def ask_comparison(self):
        """Comparison questions — tests yes/no reasoning path."""
        question = random.choice(COMPARE_WIKI + COMPARE_VECTOR)
        self._ask(question, label="comparison")

    @task(1)
    def ask_random(self):
        """Random question from full pool — general coverage."""
        question = random.choice(ALL_QUESTIONS)
        self._ask(question, label="random")

    def _ask(self, question: str, label: str = ""):
        with self.client.post(
            "/ask",
            json={"question": question},
            name=f"/ask [{label}]",   # groups requests by label in UI
            catch_response=True,
            timeout=60,               # LLM inference can be slow
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("answer"):
                    resp.failure("Empty answer returned")
                else:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def health_check(self):
        """Lightweight health check — validates server is alive."""
        self.client.get("/health", name="/health")
