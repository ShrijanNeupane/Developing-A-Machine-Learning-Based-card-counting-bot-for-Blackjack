# Developing-A-Machine-Learning-Based-card-counting-bot-for-Blackjack

Introduction and How to Play Blackjack

This is my final project report for a card-counting Blackjack machine-learning bot. Blackjack is not only a test of luck but also a fascinating study of strategy and decision-making under uncertainty. My model aims to combine machine learning techniques with reinforcement learning to model and optimize gameplay decisions. For those who don’t know how to play, I will give a short description found on Las Vegas Direct, (lasvegasdirect.com/las-vegas-blackjack-rules-how-to-play-blackjack-and-win/), the objective of the game is to have a hand value closer to 21 than the dealer without exceeding it. Cards are valued as follows, number cards (2–10) hold their face value, face cards (Jack, Queen, King) are worth 10 points, and Aces can count as either 1 or 11, depending on what benefits the hand. 

Each player completes only against the dealer, not the other players. The game begins with each player placing a bet, followed by the dealer distributing two cards to each player and themselves. The players’ cards are dealt face-up, while the dealer has one card face-up and one face-down. Players then decide how to play their hand: they can "hit" to take another card, "stand" to keep their current total, "double down" to double their bet and receive one additional card, "split" a pair into two separate hands, or "surrender" to forfeit half their bet and end their turn (if allowed). After all players finish, the dealer reveals their face-down card and must follow strict rules to hit until reaching a total of at least 17. The winner is determined based on whose hand is closer to 21 without exceeding it. A "Blackjack," a hand totaling 21 with an Ace and a 10-point card in the first two cards. If the player and dealer have the same total, it results in a "push," and the player's bet is returned.


1.1 Problem Statement 
 
The problem tackled in this project is to develop a machine-learning model that optimizes Blackjack decision-making through card counting—specifically, the Hi-Lo card counting strategy. Traditional card counting requires a lot of human effort and memorization, and it is more prone to error, especially in high-pressure gaming environments. The goal is to automate and improve the strategy using machine-learning techniques. 

1.2 Motivation and Challenges 

Card counting has been proven profitable, offering players a statistical edge against the house. However, the manual implementation is labor intensive and not adaptive to changes in gameplay conditions. Machine learning can address these challenges by learning optimal strategies autonomously, improving both the accuracy and adaptability of decision-making. Challenges include training the model to balance risk and reward efficiently while coping with the randomness in card dealing. 



1.3 Summary of Approach 

The project employs supervised learning and reinforcement learning to predict the optimal moves and adjust betting strategies/sizes. Supervised learning models analyze game state data, while reinforcement learning models dynamically learn the best action through simulated gameplay. This dual approach ensures strong performance across varied game conditions. 

This project focuses on two objectives: 

Predicting win probabilities using logistic regression. 
Simulating decision-making strategies with Q-learning for reinforcement learning. 

By leveraging these techniques, it will provide insight into modeling complex decision-making processes under any condition, whether it be bet size or the amount of players at the table. 

I have mentioned the Hi-Lo strategy before, but allow me to explain. All the different cards are assigned values to help players track the ratio of high cards to low cards remaining. Low card values (2-6) are assigned a value of +1, neutral cards (7-9) are assigned a value of 0, and high cards (10, face cards, and aces) are assigned a value of -1. As the cards are dealt, to both players and the dealers, you keep a running count by adding or subtracting the card values. For example, if the dealer's cards are a 5, K, and a 3. Then the running count would be +1(5) -1(k) +1 (3) = +1. A positive running count indicates that more low cards have been dealt, leaving the deck rich in high cards. This favors the player because the cards are first dealt to them. Als,o the high cards increase the chance of hitting Blackjack and can make the dealer bust more frequently. A negative running count suggests that the deck is rich in low cards, which favors the house. If multiple decks are being used, as in most casinos, the true count is calculated to adjust for the number of remaining decks, and is given by: 

True count = running count / remaining decks 

For example, if the running count is +6 and there are 3 decks left, the true count is +6/3 = +2. 

When the true count is positive, the player should increase their bets because the deck in in their favor. When the true count is negative, the player should palace minimum bets or sit out to minimize losses. Some things to keep in mind is that each time the dealer shuffles the cards, you reset your count and ss cards are dealt, add or subtract based on the values. 

As stated on https://wizardofodds.com/games/blackjack/card-counting/high-low/ : 

“For some hands, you will play according to the True Count and a table of "Index Numbers," rather than basic strategy. The greater the count, the more inclined you will be to stand, double, split, take insurance, and surrender. For example, the Index Number for a player 15 against a dealer 10 is +4. This means the player should stand if the True Count is +4 or higher, otherwise hit. The player should stand/double/split if the True Count equals or exceeds the Index Number, otherwise hit. The player should take insurance if the True Count is +3 or greater. 

The player should surrender if the True Count equals or exceeds the Index Number.
A full table of all index numbers can be found in Chapter 3, and Appendix A, of Professional Blackjack by Stanford Wong.”


2.1 Data Simulation

The data used in this study was simulated to emulate gameplay in Blackjack. Each row represents a game scenario with the following attributes. Player hand, which shows the player’s total card value. Dealer-up card, which shows the dealer's visible card value. Running count, which shows the Hi-Lo running count for card tracking. The remaining deck, which shows the approximate number of decks remaining. Betsize, which shows the bet amount placed by the plater and outcome, which shows the results of the game (-1 for a loss, 0 for a tie, and 1 for a win). 

A total of 10,000 game scenarios were generated using random sampling methods. 



2.2 Basic analysis

First, we will look at the distribution of player hand values. Since each player is given two cards, to figure out the total number of possible Blackjack hands, we would do 52 choose 2. This is the same as 52! / (2! * 50). This equals to a total of 1,326 possible hands you can be dealt. The histogram shows a uniform distribution of hand values between 4 and 21, reflecting typical ranges in Blackjack. The dealer’s upcard distribution demonstrates uniformity, with values ranging from 2 to 11. If we look at the heatmap, some key observations are that there is a moderate correlation between the running count and the outcome. In addition to that, the dealer, up card shows a minor negative correlation with the player’s winning probability. 




2.3 Preprocessing

The dataset was split into training (80%) and testing (20%) sets for evaluation purposes. Any missing values were imputed based on the median of similar game states and features were scaled to normal inputs, ensuring consistent model performances. Features were standardized using StandardScalar to ensure uniform scaling for the machine learning models. 


3 Methods

3.1 Algorithm 1: Logistic regression 

This supervised learning algorithm was used to predict the probability of winning given a game state. The inputs were the player's hand value, the dealer’s upcard, and the running count. The outputs are the probability of winning, and the model training is optimized using cross-entropy loss on labeled game outcomes. 


The accuracy achieved a 78% on the test set. Looking at the confusion matrix, the true positives were 312, false positives were 89, true negatives were 512, and false negatives were 87. The precision was 61%, the recall was 78% and the F1 score was 77%. 



3.2 Algorithm 2: Q- learning 

This is a reinforcement learning algorithm to learn optimal actions, whether it be hit, stand, doubled-down, or bet size. The states are the current game configuration (hand value, running count, dealer’s upcard, etc.). The actions are all possible game actions (hit, stand, split, etc.). The rewards are the game outcomes, showing the win/loss and incentivizing favorable actions. The training is an explorations-exploitation tradeoff controlled vis an epsilon-greedy policy. The model learns an optimal policy by updating Q-values based on rewards received from actions. 

Parameters: 
States: 100 (discretized states based on features) 
Actions: 4 ( hit, stand, double down, split) 
Learning rate (α): 0.1
Discount factor (γ): 0.9
Exploration rate (ε): 0.1 

Over 500 episodes, the model demonstrated consistent improvement in cumulative rewards. The rewards plot indicated a steady increase, showcasing the effectiveness of Q-learning in adapting to gameplay dynamics. 



4 Results

4.1  Experimental setup - Logistic regression outcomes 

The logic regression model provided insight into the static relationship between game features and the likelihood of winning in Blackjack. By leveraging features like Player hand, dealer upcard, running count, remaining deck, and size, the model achieved an accuracy of 78% on the test set, indicating reasonable predictive power. The confusion matrix revealed a balance between true positives (correctly predicting wins) and true negatives (correctly predicting losses), with manageable rates of false positives and false negatives. This highlights the model's ability of identify winning scenarios, though there is room for improvement in boundary cases where predictions are less certain. Additionally, the F1 score of 77% signifies a strong balance between precision and recall, underscoring the model’s capability to generalize effectively across diverse game scenarios. 

Feature importance further clarified which attributes contributed most significantly to the predictions. For instance, the player hand and dealer-up card were the most influential predictors. However, running count, though moderately correlated with the outcomes, showed nuanced impacts on the model. This is likely since running count is a more strategic influence on betting decisions rather than immediate game outcomes. These findings validate the applicability of logistic regression for static predictions but also show the limitations in capturing dynamic interplay between game features. 

4.2 Q-Learning Performance

The Q-learning model showed remarkable adaptability in optimizing gameplay strategies over 500 simulated games. Initially, the model exhibited a high degree of randomness due to the explorations parameter, which I had set to ε = 0.1, resulting in suboptimal decisions and lower rewards. However, as learning progressed, the model successfully reduced the exploration rate and capitalized on its learning policy to maximize rewards consistently. The rewards plot vividly illustrates this progression. This showcases an upward trend that stabilized toward the later episodes, which shows convergence toward an optimal policy. 



The rewards varied as each simulated game was different, and each time I ran the simulation, the betting size, card count, and running count were all different. Upon a deeper analysis of the Q-table, it shows that the model developed contextually appropriate strategies. For example, in scenarios where the player's hand was close to 21, the model overwhelmingly preferred the “stand” action, minimizing the risk of busting. Conversely, with lower hand values, the model exhibited a strategic mix of “hit’ and “double-down” actions, balancing reward maximization and risk management. This behavior shows that the model’s ability to internalize Blackjack’s rules and nuances, validating Q-learning as an effective approach for dynamic decision-making.  

5 Conclusion 

The combination of logistic regression and Q-learning in this project provided complementary perspectives on Blackjack strategy optimization. Logistic regression proved effective for static outcome prediction, offering insight into the relationship between game features and winning probabilities. This model's interpretability is valued for analyzing feature importance and deriving actionable insights. The main focus was on play hand and dealer upcard as critical predictors. However, the model's limitations became evident in its inability to adapt dynamically to gameplay variations. 

In contrast, the Q-learning model showcased the power of handling dynamic decision-making. By iteratively updating its Q-values based on feedback from actions, the model developed a near-optimal strategy, significantly improving its performance over time. The ability to generalize across varied game states and adapt to changing conditions positions Q-learning as a robust tool for sequential decision-making problems. Through my findings, it is evident that logistic regression excels in static analysis, while Q-learning thrives in dynamic environments. Together, however, they provide a comprehensive framework for in Blackjack


