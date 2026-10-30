# Exploratory Data Analysis

## 1. Matches per year

Matches are exponentially increasing from 2005 when this format was introduced

## 2. Average first-innings score per year

Avg 1st innings scores is flucating and was the lowest in 2021

## 3. Venue scoring differences

Average First-Innings Score at 10 Busiest Venues is highest in Wankhede Stadium and lowest in Bayuemas Oval 

## 4. Venue chase win rates

Chase Win Rate at 10 Busiest Venues is highest in Sawai Mansingh Stadium and lowest in 
MA Chidambaram Stadium

## 5. Scoring rate by innings phase

Scoring rate is the highest in the last 5 overs

## 6. Chase win-rate heatmap

Chase win probability is strongly influenced by the runs required and balls remaining. Teams generally have a higher chance of winning when they need fewer runs and have more deliveries available.

## 7. First-innings projection

There is a strong positive relationship between the score after 10 overs and the final first-innings total.


## Modelling implications
- Cricket scoring patterns may change across years.
- Venue conditions may influence match outcomes.
- Scoring rates vary across innings phases.
- The model should consider the current match situation.
- Include runs required and balls remaining in the chase model.
- Investigate venue and year as potential features.
- Account for different scoring rates across innings phases.
- Evaluate uncertainty in first-innings score predictions.
- Use match-based train/test splitting to prevent data leakage.