function [plPar,plRes]=PLA(func,par,i,maxPar,threshold,lb,ub,Optimoptions,minStep,maxStep,minChange,maxChange,nr)
% PLA Profile Likelihood Analysis calculates profile likelihood along parameter nr. i
% - in increasing direction if maxPar > par, with par the calibrated parameter value
% - in decreasing direction if maxPar < par
%
%Assume prefdenotes a calibrated set of parameter of the model. 
%Beginning from pref(i), sample along the profile likelihood in either  
%increasing or decreasing direction of pref(i), by following procedure:
%a. Take a step pstep in either direction of pref(i);
%b. Re-optimize all pref(j~=i) ;
%Repeat the above steps until
%   1) the threshold (based on chi-squared) is exceeded
%   2) or the maximum number of steps nr is reached
%   3) or the boundary of parameter region is reached
%depending on the computational condition
%pstep is chosen in an adaptive manner, taking large steps if the likelihood
%is flat and small steps if the change of the likelihood is steep.
%
%   [plPar,plRes]=PLA(func,par,i,maxPar,threshold,lb,ub,Optimoptions,
%                       minStep,maxStep,minChange,maxChange,nr)
%
% % INPUT
%   - func      : function handle referring to cost function for model fitting (lsqnonlin) 
%   - par       : calibrated parameters
%   - i         : the i-th parameter for PL calculation
%   - maxPar    : max value of i-th parameter
%   - threshold : threshold - chi square distribution
%   - lb        : lower bounds for parameter estimation (lsqnonlin)
%   - ub        : upper bounds for parameter estimation (lsqnonlin)
%   - Optimoptions : options for optimization
%   - minStep   : minimal step factor
%   - maxStep   : maximal step factor
%   - minChange : minimal change of resnorm
%   - maxChange : maximal change of resnorm
%   - nr        : no. of samples in profile likelihood
%
% % OUTPUT
%   - plPar     : vector of values for i-th parameter 
%   - plRes     : vector of corresponding resnorm values (profile likelihood)
%
% [plPar,plRes]=PLA(func,par,i) uses default values for inputs:
% maxPar = 10*par(i), threshold = chi2inv(0.5,size(par)),
% lb=[], ub=[], Optimoptions=[], minStep=0.01, maxStep, 
% minChange=0.001, maxChange=0.05, nr=100

%History
%02-Mar-2011 Natal van Riel, TU/e

disp(' ');disp(['Profile Likelihood calculation for parameter ' int2str(i)]);disp(' ');

% set default values
if nargin < 4, maxPar = 10*par(i);end
if nargin < 5, threshold = chi2inv(0.5,size(par));end   
if nargin < 6, lb=[]; end
if nargin < 7, ub=[]; end
if nargin < 8, Optimoptions=[]; end
if nargin < 9, minStep=0.01; end
if nargin < 10, maxStep=0.1; end
if nargin < 11, minChange=0.001; end
if nargin < 12, maxChange=0.05; end
if nargin < 13, nr=100; end

minChange = minChange * threshold;
maxChange = maxChange * threshold;
% %----Choice 1----------%
% minStep = par(i)*minStep;
% maxStep = par(i)*maxStep;
% %----Choice 2---------%
minStep = abs(par(i)-maxPar)*minStep;
maxStep = abs(par(i)-maxPar)*maxStep;

step = minStep;

% specify the direction: 1 increase -1 decrease
flag = sign(maxPar - par(i));

% select parameters to be re-optimized
nPar = [par(1:i-1) par(i+1:end)];
if isempty(lb), lb=[]; else lb = [lb(1:i-1) lb(i+1:end)]; end
if isempty(ub), ub=[]; else ub = [ub(1:i-1) ub(i+1:end)]; end

% function handle for optimization
optFun = @(nPar,iPar)feval(func,[nPar(1:i-1) iPar nPar(i:end)]); 
res    = sum(func(par).^2);

k = 1;
count = 1;
plPar(1) = par(i);
plRes(1) = res;

while ((plPar(end)<=maxPar && logical(flag+1) && count<=nr) || (plPar(end)>=maxPar && ~logical(flag+1) && count<=nr))
    % take a step along the parameter 
    tempPar = flag*step+plPar(end);
    
    [x,resnorm] = lsqnonlin( @(nPar)optFun(nPar,tempPar),nPar(k,:),lb,ub,Optimoptions);
    
    % change of the resnorm value
    resChange = resnorm - plRes(end);
%     if resnorm > threshold
%         break;
%     end
    if resnorm - res>threshold
        break;
    end
     
    if resChange > maxChange; 
        step = step/2; 
        step = max(step,minStep);
        continue;
    else
        if resChange < minChange
            step = step*2;
            step = min(step,maxStep);
        end
            count = count+1;
        k = k+1;
        plPar(k) = tempPar; 
        plRes(k) = resnorm;
        nPar(k,:)= x;
    end
end
