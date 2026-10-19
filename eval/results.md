Model: `gpt-4.1-mini` · MIN_SCORE = 0.35 · top_k = 4

| # | Question | Type | Answered | Correct page in sources | Fact in answer | Top score | Latency |
|---|---|---|---|---|---|---|---|
| 1 | How many paid vacation days do full-time employees get per year? | answerable · direct | ✅ | ✅ | ✅ | 0.707 | 1090 ms |
| 2 | How many unused vacation days can be carried over, and when do they expire? | answerable · direct | ✅ | ✅ | ✅ | 0.523 | 1178 ms |
| 3 | How long is paid parental leave? | answerable · direct | ✅ | ✅ | ✅ | 0.577 | 933 ms |
| 4 | How many days per week can I work from home? | answerable · direct | ✅ | ✅ | ✅ | 0.471 | 1037 ms |
| 5 | What is the monthly internet allowance for remote employees? | answerable · direct | ✅ | ✅ | ✅ | 0.535 | 1086 ms |
| 6 | What is the hotel limit per night in London? | answerable · direct | ✅ | ✅ | ✅ | 0.391 | 1224 ms |
| 7 | When am I allowed to fly business class? | answerable · direct | ✅ | ✅ | ✅ | 0.438 | 1522 ms |
| 8 | Within how many days do I have to submit an expense? | answerable · direct | ✅ | ✅ | ✅ | 0.536 | 1024 ms |
| 9 | What is the minimum password length? | answerable · direct | ✅ | ✅ | ✅ | 0.442 | 1020 ms |
| 10 | How quickly must I report a lost or stolen laptop? | answerable · direct | ✅ | ✅ | ✅ | 0.359 | 1037 ms |
| 11 | How long is the probation period? | answerable · direct | ✅ | ✅ | ✅ | 0.652 | 1055 ms |
| 12 | How big is the home office budget for new employees? | answerable · direct | ✅ | ✅ | ✅ | 0.490 | 1204 ms |
| 13 | Am I allowed to do my job from a different European country for a while? | answerable · paraphrase | ✅ | ✅ | ✅ | 0.521 | 1053 ms |
| 14 | How much will the company pay for my overnight stay when I travel to Paris? | answerable · paraphrase | ✅ | ✅ | ✅ | 0.488 | 1035 ms |
| 15 | If my work computer is taken from me, who do I tell and how fast? | answerable · paraphrase | ✅ | ✅ | ✅ | 0.398 | 1237 ms |
| 16 | Do I need a medical certificate if I'm off ill for a while? | answerable · paraphrase | ✅ | ✅ | ✅ | 0.438 | 1122 ms |
| 17 | What money do new hires get for equipment, and what do remote employees get every month? | answerable · multi-section | ✅ | ✅ | ✅ | 0.476 | 1302 ms |
| 18 | How far ahead must I request vacation, and within how many days must I file a travel expense? | answerable · multi-section | ✅ | ✅ | ✅ | 0.477 | 1205 ms |
| 19 | How much notice is needed for vacation longer than ten working days, and how long does probation last? | answerable · multi-section | ✅ | ✅ | ✅ | 0.600 | 1740 ms |
| 20 | How much is reimbursed per kilometre when I use my own car for work? | answerable · exact-value | ✅ | ✅ | ✅ | 0.596 | 1123 ms |
| 21 | By what date must the annual security training be completed? | answerable · exact-value | ✅ | ✅ | ✅ | 0.423 | 1059 ms |
| 22 | What is the CEO's salary? | unanswerable · off-topic | refused ✅ (layer 1) | – | – | 0.270 | 418 ms |
| 23 | What's the weather in Paris? | unanswerable · off-topic | refused ✅ (layer 1) | – | – | 0.189 | 285 ms |
| 24 | What dental insurance plan does Acme offer? | unanswerable · off-topic | refused ✅ (layer 2) | – | – | 0.378 | 836 ms |
| 25 | Does Acme provide company cars to employees? | unanswerable · off-topic | refused ✅ (layer 2) | – | – | 0.567 | 1163 ms |
| 26 | What is Acme Corp's current stock price? | unanswerable · off-topic | refused ✅ (layer 2) | – | – | 0.515 | 1193 ms |
| 27 | Who won the 2022 FIFA World Cup? | unanswerable · off-topic | refused ✅ (layer 1) | – | – | 0.046 | 272 ms |
| 28 | How many paid sick days do employees get per year? | unanswerable · near-miss | refused ✅ (layer 2) | – | – | 0.617 | 950 ms |
| 29 | How often do passwords have to be changed? | unanswerable · near-miss | refused ✅ (layer 2) | – | – | 0.481 | 1026 ms |
| 30 | How much paid leave do I get when a close relative dies? | unanswerable · near-miss | refused ✅ (layer 2) | – | – | 0.411 | 946 ms |
| 31 | What is the monthly wellness or gym allowance? | unanswerable · near-miss | refused ✅ (layer 2) | – | – | 0.362 | 862 ms |

| Metric | Result |
|---|---|
| Retrieval hit rate (all expected pages in sources) | 100% (21/21) |
| Answerable questions answered | 100% (21/21) |
| Answers containing the expected fact | 100% (21/21) |
| Unanswerable questions correctly refused | 100% (10/10) |
| Median latency | 1055 ms |
