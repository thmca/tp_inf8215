include "globals.mzn";

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

predicate is_in_bound(var int: val_x, var int: val_y, var int :min_bound_x, var int: max_bound_x, var int :min_bound_y, var int: max_bound_y) = 
  val_x <= max_bound_x /\ val_x > min_bound_x /\ val_y <= max_bound_y /\ val_y > min_bound_y;
  
function var int: is_in_x(int: x, int: y) = 
  sum(n in 1..n_object)(n*is_in_bound(x,y,shape_offset_x[n], shape_offset_x[n] + shape[n,1], shape_offset_y[n], shape_offset_y[n] + shape[n,2]));
  
constraint end_x <= max_x /\ end_y <= max_y;

constraint forall(n in 1..n_object)(shape[n,1]+shape_offset_x[n] <= end_x);
constraint forall(n in 1..n_object)(shape[n,2]+shape_offset_y[n] <= end_y);


%constraint forall(n in 1..n_object)(forall(i in 1..shape[n,1])(forall(j in 1..shape[n,2])( 
 % n = room[i + shape_offset_x[n],j+shape_offset_y[n]])));
  
%constraint forall(i in 1..max_length)(forall(j in (end_y+1)..max_width)( 
 % room[i,j] = 0));
  
%constraint forall(i in (end_x+1)..max_length)(forall(j in 1..max_width)( 
 %  room[i,j] = 0));

constraint forall(i in 1..end_x)(forall(j in 1..end_y)( 
  room[i,j] = is_in_x(i,j)));
  



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