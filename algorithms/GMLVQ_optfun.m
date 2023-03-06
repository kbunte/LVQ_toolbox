function [f,G]  = GMLVQ_optfun(variables,training_data,LabelEqualsPrototype,LRrelevances,LRprototypes,prototypeLabel,regularization)
% [f G] = GMLVQ_optfun(variables) 
% function to be optimzed by matrix relevance learning vector quantization
% variables = [prototype matrix;omega matrix]
% global variables are
%   training_data        : data vectors as row vectors, i.e. attributes in columns
%   LabelEqualsPrototype : binary matrix indicating coocurrences of
%   training labels and prototype labels
%   prototypeLabel       : label vector for the prototypes
%   regularization       : the regularization parameter
%   LRrelevances         : learning rate for the relevance matrix
%   LRprototypes         : learning rate for the prototypes
%
% Kerstin Bunte (modified based on the code of Marc Strickert http://www.mloss.org/software/view/323/)
% kerstin.bunte@googlemail.com
% Fri Nov 09 14:13:52 CEST 2012
%
% Conditions of GNU General Public License, version 2 apply.
% See file 'license-gpl2.txt' enclosed in this package.
% Programs are not for use in critical applications!
% 
if isempty(LRprototypes) % values between 1e-2,1e-3,... 1e-8 seem pragmatic
    LRprototypes = 1; % no relevance learning by default
end
if isempty(LRrelevances) % values between 1e-2,1e-3,... 1e-8 seem pragmatic
    LRrelevances = 0; % no relevance learning by default
end
[n_data, n_dim] = size(training_data);
nb_prototypes =  numel(prototypeLabel);
omegaT = variables(nb_prototypes+1:end,:)';
n_vec = size(variables,1) - nb_prototypes;

dists = squaredEuclidean(training_data*omegaT, variables(1:nb_prototypes,:)*omegaT);

Dwrong = dists;
Dwrong(LabelEqualsPrototype) = realmax(class(Dwrong));   % set correct labels impossible
[distwrong,pidxwrong] = min(Dwrong.'); % closest wrong
clear Dwrong;

Dcorrect = dists;
Dcorrect(~LabelEqualsPrototype) = realmax(class(Dcorrect)); % set wrong labels impossible
[distcorrect,pidxcorrect] = min(Dcorrect.'); % closest correct
clear Dcorrect;

distcorrectpluswrong = distcorrect + distwrong;
distcorrectminuswrong = distcorrect - distwrong;
mu = distcorrectminuswrong ./ distcorrectpluswrong;
% callitq = 1./(1 + exp(-squashsigmoid * mu)); % apply sigmoidal

if regularization
    regTerm = regularization * log(det(omegaT'*omegaT));
else
    regTerm = 0;
end
% sum(diag(omega'*omega))
normTerm = (1-sum(arrayfun(@(d) sum(omegaT(d,:).*omegaT(d,:)),1:size(variables,2))))^2;
f = mean(mu) - regTerm + normTerm; % disp([mean(mu) regTerm]);
% f = mean(callitq);

if nargout > 1  % gradient needed not just function eval
    G = zeros(size(variables)); % initially no gradient
    %       callitq = squashsigmoid * callitq .* (1-callitq); % derivative of sigmoid
    %       distcorrectpluswrong = 2 * callitq ./ distcorrectpluswrong.^2; % degeneration?
    distcorrectpluswrong = 4 ./ distcorrectpluswrong.^2; % norm_factor for derivative for every data sample
    if LRrelevances > 0
        Gw = zeros(n_vec,n_dim);
    end
    for k=1:nb_prototypes%(n_vec+1):size(lambda,1) % update all prototypes        
        idxc = (k == pidxcorrect);  % Js: idxs where actual prototype is nearest correct
        idxw = (k == pidxwrong);    % Ks: idxs where actual prototype is nearest wrong

        dcd =  distcorrect(idxw) .* distcorrectpluswrong(idxw);
        dwd =    distwrong(idxc) .* distcorrectpluswrong(idxc);
        if LRrelevances > 0
            % part of derivative of distance
            difc = bsxfun(@minus,training_data(idxc,:),variables(k,:)); % DJs
            difw = bsxfun(@minus,training_data(idxw,:),variables(k,:)); % DKs
            % update omega          
            Gw = Gw - (bsxfun(@times,difw,dcd.') * omegaT).' * difw + ...
                      (bsxfun(@times,difc,dwd.') * omegaT).' * difc;
            if LRprototypes > 0
                G(k,:) = dcd * difw - dwd * difc;
            end
        else
            if LRprototypes > 0
                G(k,:) = dcd * training_data(idxw,:) - dwd * training_data(idxc,:) + (sum(dwd)-sum(dcd)) * variables(k,:);
            end
        end
    end
if regularization
    f3 = (pinv(omegaT'))';                
else
    f3 = 0;
end  
    % some rescalings needed
    if LRrelevances > 0
        normOterm = -4*(1-sum(arrayfun(@(d) sum(omegaT(d,:).*omegaT(d,:)),1:size(omegaT,1)))).*omegaT';
        G(nb_prototypes+1:nb_prototypes+n_vec,:) = 2/n_data * LRrelevances * Gw - regularization*f3 + normOterm;
    end
    if LRprototypes > 0
        G(1:nb_prototypes,:) = 1./n_data * LRprototypes * G(1:nb_prototypes,:) * omegaT * omegaT.';
    end
%     G = G .* (1 + .0001 * (rand(size(G))-.5)); % help break symmetries
end
% if 0,
% w = variables(1:nb_prototypes,:);
% dJs = zeros(1,nb_samples);
% dKs = zeros(1,nb_samples);
% Js = zeros(1,nb_samples);
% Ks = zeros(1,nb_samples);
% norm_factors = zeros(1,nb_samples);
% DJs = zeros(nb_samples,size(training_data,2));
% DKs = zeros(nb_samples,size(training_data,2));
% for i=1:nb_samples
%     % select one training sample randomly
%     xi = training_data(i,:);
%     c_xi = trainLab(i);
% 
%     dist = ((xi(ones(size(w,1),1),:))-w)*omegaT*(omegaT'*((xi(ones(size(w,1),1),:))-w)');
%     dist = diag(dist);
%     % determine the two winning prototypes
%     % nearest prototype with the same class
%     [sortDist,sortIdx] = sort(dist);
%     count = 1;
%     J = sortIdx(count);
%     while prototypeLabel(sortIdx(count)) ~= c_xi, 
%         count = count+1;
%         J = sortIdx(count);
%     end
%     dJ = sortDist(count);
%     dJs(i) = dJ;
%     Js(i) = J;
%     count = 1;
%     K = sortIdx(count);
%     while prototypeLabel(sortIdx(count)) == c_xi, 
%         count = count+1;
%         K = sortIdx(count);
%     end
%     dK = sortDist(count);
%     dKs(i) = dK;
%     Ks(i) = K;
% 
%     wJ = w(J,:);
%     wK = w(K,:);
%     % prototype update
%     norm_factors(i) = 4/((dJ + dK)^2);
%     DJ = (xi-wJ);
%     DK = (xi-wK);
%     DJs(i,:) = DJ;
%     DKs(i,:) = DK;
% end
% end