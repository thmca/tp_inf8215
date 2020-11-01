
include "globals.mzn"; 
   
% Parameters of the problems
int: n = 5;
array[1..n] of int: left_vision = [2,1,3,2,3];
array[1..n] of int: right_vision = [3,4,1,3,2];
array[1..n] of int: up_vision = [2,1,3,2,2];
array[1..n] of int: down_vision = [3,4,2,1,2];

% Variable: value for each cell
array[1..n,1..n] of var 1..n: city;

predicate right_vision_check(int: x, int: i) =
forall(k in i+1..n)(
  city[i,x]>city[k,x]);
  
predicate left_vision_check(int: x, int: i) =
forall(k in i+1..n)(
  city[n-i+1,x]>city[n-k+1,x]);
  
predicate down_vision_check(int: x, int: i) =
forall(k in i+1..n)(
  city[x,i]>city[x,k]);
  
predicate up_vision_check(int: x, int: i) =
forall(k in i+1..n)(
  city[x, n-i+1]>city[x, n-k+1]);


% First constraint
constraint forall(j in 1..n)(alldifferent(i in 1..n)(city[i, j]));
constraint forall(j in 1..n)(alldifferent(i in 1..n)(city[j, i]));

%On regarde le nombre de building pour lesquel il y a aucun building plus grand apres lui


constraint forall(x in 1..n)(
  right_vision[x]==sum(i in 1..n)(right_vision_check(x,i))
  /\ left_vision[x]==sum(i in 1..n)(left_vision_check(x,i))
  /\ down_vision[x]==sum(i in 1..n)(down_vision_check(x,i))
  /\ up_vision[x]==sum(i in 1..n)(up_vision_check(x,i)));


%constraint forall(x in 1..n)(left_vision[x]==sum(i in 1..n)(left_vision_check(x,i)));


%constraint forall(x in 1..n)(down_vision[x]==sum(i in 1..n)(down_vision_check(x,i)));


%constraint forall(x in 1..n)(up_vision[x]==sum(i in 1..n)(up_vision_check(x,i)));




output
["city:"]++
[
  if j = 1 then "\n" else " " endif ++
    show(city[j,i])
  | i in 1..n, j in 1..n
];
