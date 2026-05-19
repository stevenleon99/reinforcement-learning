**Math Expression**
```
A(s, a) = Q(s, a) - V(s)
```

**Meaning**
Measure how much better of the action than what we normally expect from this state

**Example**
```
Suppose the agent is in one state:
state = ball is coming toward the paddle


The average value of this state is:
V(s) = 5


Now compare possible actions:
Q(s, LEFT)  = 8Q(s, RIGHT) = 3Q(s, FIRE)  = 5


Then:
A(s, LEFT)  = 8 - 5 = +3
A(s, RIGHT) = 3 - 5 = -2
A(s, FIRE)  = 5 - 5 = 0


Interpretation:
LEFT is better than average in this state.RIGHT is worse than average.FIRE is about average.
```


**Application**
Usually we can make this strategy:
If A(s, a) > 0:  
increase probability of taking action a in state s  
If A(s, a) < 0:  
decrease probability of taking action a in state s


