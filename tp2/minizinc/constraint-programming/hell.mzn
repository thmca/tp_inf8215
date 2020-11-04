include "globals.mzn";

% A rouler avec le Solver Chuffed 0.10.4

int: n_object;
array[1..n_object,1..2] of int: shape; % (x,y) shape of each object
int: max_length;
int: max_width;

var 1..max_width: end_x;
var 1..max_width: end_y;

array[1..max_length, 1..max_width] of var 0..n_object: room;
array[1..n_object] of var 0..max_length: shape_offset_x;
array[1..n_object] of var 0..max_width: shape_offset_y;

int: max_x = sum(n in 1..n_object)(shape[n,1]);
int: max_y = sum(n in 1..n_object)(shape[n,2]);
  
constraint end_x <= max_x /\ end_y <= max_y;

constraint forall(n in 1..n_object)(shape[n,1]+shape_offset_x[n] <= end_x /\ shape[n,2]+shape_offset_y[n] <= end_y);
%constraint forall(n in 1..n_object)(shape[n,2]+shape_offset_y[n] <= end_y);


constraint forall(n in 1..n_object)(forall(i in 1..shape[n,1])(forall(j in 1..shape[n,2])( 
  n = room[i + shape_offset_x[n],j+shape_offset_y[n]])));
  

solve minimize end_x*end_y;


output
["Nouvelle meilleure solution trouvée: \n"]++
["Surface totale occupée dans l'entrepôt: "]++[show(end_x*end_y)]++["\n"]++
["Longueur (x): "]++[show(end_x)]++["\n"]++
["Largeur (y): "]++[show(end_y)]++["\n"]++
["Disposition des boîtes selon le coin inférieur gauche (x,y):"]++
[
  "\n Boîte " ++ show(i) ++ ": " ++ "("++
    show(shape_offset_x[i]) ++ "," ++ show(shape_offset_y[i]) ++")"
  | i in 1..n_object
];