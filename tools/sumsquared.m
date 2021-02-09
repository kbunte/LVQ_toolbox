function sq  = sumsquared(x,dim) 
% sq  = sumsquared(x,dim) 
% sum of all squared element of matrix x along dimension dim

persistent isoctave

if isempty(isoctave)
  isoctave = exist('OCTAVE_VERSION','builtin');
end

if nargin == 1
  dim = 1;
end

if isoctave
  sq = sumsq(x,dim);
else
  sq = sum(x.^2, dim);
end

