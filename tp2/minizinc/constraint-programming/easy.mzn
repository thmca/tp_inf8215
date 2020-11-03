include "globals.mzn";

int: n; 
array [1..n] of var 1..n: d; % One variable for each column. The domain corresponds to the cell in the column where the drone must be located

%checks the vertical and horizontal lines
constraint all_different(d);
%checks the diagonals
constraint forall(i,j in 1..n)(i == j \/ (((j-i + d[i]) != d[j] /\ (d[i] - (j-i) != d[j]))));

solve satisfy;
output	
  [if fix(d[i]) = j then "D " 
   else ". " 
   endif
      
   ++
   
   if j = n then "\n" 
   else "" endif
   
   | i, j in 1..n];