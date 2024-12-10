import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
from sklearn.metrics import precision_score

y_true = np.random.randint(0, 2, size=100)  # Ground truth
y_pred = np.random.randint(0, 2, size=100)  # Predictions

precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
print("Precision:", precision)


# Step 1: Data Simulation
np.random.seed(42)
random.seed(42)

def generate_blackjack_data(n_games):
    data = []
    for _ in range(n_games):
        player_hand = np.random.randint(4, 22)  # Possible hand values
        dealer_upcard = np.random.randint(2, 12)  # Dealer's upcard (2-11)
        running_count = np.random.randint(-10, 11)  # Hi-Lo running count
        remaining_deck = np.random.randint(1, 5)  # Approximate deck composition indicator
        bet_size = np.random.randint(1, 101)  # Bet size (1-100)
        outcome = np.random.choice([1, 0, -1], p=[0.4, 0.4, 0.2])  # Win, Tie, Loss

        data.append([player_hand, dealer_upcard, running_count, remaining_deck, bet_size, outcome])
    return pd.DataFrame(data, columns=["PlayerHand", "DealerUpcard", "RunningCount", "RemainingDeck", "BetSize", "Outcome"])

data = generate_blackjack_data(10000)

# Step 2: Data Exploration and Visualization
print("Dataset Overview:\n", data.head())

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
data['PlayerHand'].hist(bins=18, color='blue', alpha=0.7)
plt.title("Distribution of Player Hand Values")
plt.xlabel("Player Hand Value")

plt.subplot(1, 2, 2)
data['DealerUpcard'].hist(bins=10, color='green', alpha=0.7)
plt.title("Distribution of Dealer Upcards")
plt.xlabel("Dealer Upcard")

plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Step 3: Data Preprocessing
# Scale features
scaler = StandardScaler()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

X = data.drop("Outcome", axis=1)
y = data["Outcome"].apply(lambda x: 1 if x == 1 else 0)  # Binary classification (win vs. not win)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))

# Step 5: Q-Learning Implementation
class BlackjackQLearning:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((states, actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action])

# Simulate Q-Learning
states = 100  # Discretized states
actions = 4  # Hit, Stand, Double Down, Split

q_learning_agent = BlackjackQLearning(states, actions)

def simulate_q_learning(agent, episodes):
    rewards = []
    for _ in range(episodes):
        state = np.random.randint(0, states)
        total_reward = 0
        for _ in range(10):  # 10 steps per game
            action = agent.choose_action(state)
            reward = np.random.choice([-1, 0, 1])  # Simulated reward
            next_state = np.random.randint(0, states)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
    return rewards

rewards = simulate_q_learning(q_learning_agent, 500)

plt.plot(rewards)
plt.title("Q-Learning: Rewards Over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.show()
