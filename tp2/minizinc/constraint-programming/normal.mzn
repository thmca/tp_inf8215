include "globals.mzn"; 

int: n;
array[1..n, 1..n] of var 0..1: grid; % var is used in order to take into account the "not-bounded case" _
 
% decision variables: mine or not on the specific cell
array[1..n, 1..n] of var 0..1: x;

% constraint forall(i, j in 1..n) (x[i,j] =1);
% initializing the grid: '_' values on the grid correspond to variables not already set (i.e. the choices you have to make)
constraint forall(i in 1..n, j in 1..n)(x[i,j] = grid[i,j]);

%Assure que tous les colonnes on que deux cases consecutives avec et sans mines
constraint forall(i,j in 1..n where j+1 <= n /\ j+2 <= n)(if x[i,j] == 0 /\ x[i,j+1] == 0 then x[i,j+2] = 1 endif);
constraint forall(i,j in 1..n where j+1 <= n /\ j+2 <= n)(if x[i,j] == 1 /\ x[i,j+1] == 1 then x[i,j+2] = 0 endif);

%Assure que tous les lignes on que deux case consecutive avec et sans mines
constraint forall(i,j in 1..n where i+1 <= n /\ i+2 <= n)(if x[i,j] == 0 /\ x[i+1,j] == 0 then x[i+2,j] = 1 endif);
constraint forall(i,j in 1..n where i+1 <= n /\ i+2 <= n)(if x[i,j] == 1 /\ x[i+1,j] == 1 then x[i+2,j] = 0 endif);

%Cette ligne permet de conter le nombre de mine sur la premiere ligne pour comparer avec les autres
var int: sum_mines;

%Assurence que tous les lignes et colonnes contiennent le meme nombres de mines
constraint forall(i in 1..n) (sum(j in 1..n) (x[i,j]) == sum_mines);
constraint forall(i in 1..n) (sum(j in 1..n) (x[j,i]) == sum_mines);

array[int] of var int:row1 = [x[1,i] | i in 1..n];
array[int] of var int:column2 = [x[2,i] | i in 1..n];

constraint forall(i,j in 1..n where i != j )(row(x,i)!=row(x,j) /\ col(x,i)!=col(x,j));

 
solve satisfy;

output [
   if j = 1 then "\n" else " " endif ++
     show(x[i,j])
   | i,j in 1..n
];