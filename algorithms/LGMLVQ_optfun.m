function [f,G]  = LGMLVQ_optfun(variables,training_data,LabelEqualsPrototype,LRrelevances,LRprototypes,prototypeLabel,nb_dims,regularization)
% [f G] = LGMLVQ_optfun(variables) 
% function to be optimzed by Localized Generalized Matrix relevance Learning Vector Quantization
% variables = [prototype matrix;omega matrix]
% global variables are
%   training_data        : data vectors as row vectors, i.e. attributes in columns
%   LabelEqualsPrototype : binary matrix indicating coocurrences of
%   training labels and prototype labels
%   prototypeLabel       : label vector for the prototypes
%   nb_dims              : number of dimension for each matrix
%   regularization       : the regularization parameters
%   LRrelevances         : learning rate for the relevance matrix
%   LRprototypes         : learning rate for the prototypes
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
[nb_samples, nb_features] = size(training_data);
nb_prototypes =  numel(prototypeLabel);
classes = unique(prototypeLabel);
if numel(prototypeLabel)~=numel(nb_dims)
    classwise = 1;
else
    classwise = 0;
end
% omegaT = variables(nb_prototypes+1:end,:)';
psis = variables(nb_prototypes+1:end,:);
model = struct('w',variables(1:nb_prototypes,:),'c_w',prototypeLabel,'psis',[]);
model.psis = mat2cell(psis,nb_dims,nb_features)';

dists = computeDistance(training_data,model.w,model);
% dists = squaredEuclidean(training_data*omegaT, variables(1:nb_prototypes,:)*omegaT);

Dwrong = dists;
Dwrong(LabelEqualsPrototype) = realmax(class(Dwrong));   % set correct labels impossible
[distwrong, pidxwrong] = min(Dwrong.'); % closest wrong
clear Dwrong;

Dcorrect = dists;
Dcorrect(~LabelEqualsPrototype) = realmax(class(Dcorrect)); % set wrong labels impossible
[distcorrect, pidxcorrect] = min(Dcorrect.'); % closest correct
clear Dcorrect;

distcorrectpluswrong = distcorrect + distwrong;
distcorrectminuswrong = distcorrect - distwrong;
mu = distcorrectminuswrong ./ distcorrectpluswrong;
% callitq = 1./(1 + exp(-squashsigmoid * mu)); % apply sigmoidal
% f = mean(callitq);
if sum(regularization)>0    
    regTerm = regularization .* cellfun(@(matrix) log(det(matrix*matrix')),model.psis);
    f = sum(mu-1/nb_samples*regTerm(pidxcorrect)-1/nb_samples*regTerm(pidxwrong));
%     f = sum(mu-regTerm(pidxcorrect)-regTerm(pidxwrong));
else
    f = sum(mu);
end

if nargout > 1  % gradient needed not just function eval    
%     if 1,
    G = zeros(size(variables)); % initially no gradient
    %       callitq = squashsigmoid * callitq .* (1-callitq); % derivative of sigmoid
    %       distcorrectpluswrong = 2 * callitq ./ distcorrectpluswrong.^2; % degeneration?    
%     distcorrectpluswrong = 4 ./ distcorrectpluswrong.^2; % norm_factor for derivative for every data sample
    normfactors = 4 ./ distcorrectpluswrong.^2;
    if LRrelevances > 0
        Gw = arrayfun(@(x) zeros(size(model.psis{x})), 1:length(model.psis),'uniformoutput',0);
    end
    for k=1:nb_prototypes%(n_vec+1):size(lambda,1) % update all prototypes        
        idxc = (k == pidxcorrect);  % Js: idxs where actual prototype is nearest correct
        idxw = (k == pidxwrong);    % Ks: idxs where actual prototype is nearest wrong
        if classwise, 
            rightIdx = find(model.c_w(k)==classes,1);            
        else
            rightIdx = k;
        end
        dcd =  distcorrect(idxw) .* normfactors(idxw); % 4*dJ/(dJ+dK)^2 where actual w is nearest wrong
        dwd =    distwrong(idxc) .* normfactors(idxc); % 4*dK/(dJ+dK)^2 where actual w is nearest correct
        
        % part of derivative of distance
        difc = bsxfun(@minus,training_data(idxc,:),variables(k,:)); % DJs
        difw = bsxfun(@minus,training_data(idxw,:),variables(k,:)); % DKs
        if LRprototypes > 0
%             G(k,:) = dcd * training_data(idxw,:) - dwd * training_data(idxc,:) + (sum(dwd)-sum(dcd)) * variables(k,:);                
% dwJ = (2*dK/norm_factor)*2*model.psis{rightIdxJ}'*model.psis{rightIdxJ}*DJ';
% dwK = (2*dJ/norm_factor)*2*model.psis{rightIdxK}'*model.psis{rightIdxK}*DK';                
% model.w(J,:) = wJ + alphas(epoch) * dwJ';
% model.w(K,:) = wK - alphas(epoch) * dwK';
% dwJ1 = (normJforK(find(Ks==1,1))*model.psis{rightIdx}'*model.psis{rightIdx}*DKs(find(Ks==1,1),:)')'
% sumdwK = normJforK(find(Ks==1))*DKs(find(Ks==1),:)*model.psis{rightIdx}'*model.psis{rightIdx}
% sumdwJ = normKforJ(find(Js==1))*DJs(find(Js==1),:)*model.psis{rightIdx}'*model.psis{rightIdx}
% dwJ1 = dcd(1)*difw(1,:)*model.psis{rightIdx}'*model.psis{rightIdx}
% sumdwK = dcd*difw*model.psis{rightIdx}'*model.psis{rightIdx}
% sumdwJ = dwd*difc*model.psis{rightIdx}'*model.psis{rightIdx}
            G(k,:) = (dcd*difw - dwd*difc)*model.psis{rightIdx}'*model.psis{rightIdx};
        end
        if LRrelevances > 0            
            % update omega          
%             Gw = Gw - (bsxfun(@times,difw,dcd.') * omegaT).' * difw + ...
%                       (bsxfun(@times,difc,dwd.') * omegaT).' * difc;
%             if sum(regularization)>0,
%                 f3 = (pinv(model.psis{rightIdx}))';                
%             else
%                 f3 = 0;
%             end  
%             f1 =  (2*dK/norm_factor)*2*(model.psis{rightIdxJ}*DJ'*DJ);
%             f2 = (-2*dJ/norm_factor)*2*(model.psis{rightIdxK}*DK'*DK);
%             model.psis{rightIdxJ} = model.psis{rightIdxJ}-epsilons(epoch) * (f1 - regularization(rightIdxJ) * f3J);
%             model.psis{rightIdxK} = model.psis{rightIdxK}-epsilons(epoch) * (f2 - regularization(rightIdxK) * f3K);
            Gw{rightIdx} = Gw{rightIdx} - (bsxfun(@times,difw,dcd.') * model.psis{rightIdx}').' * difw + ...
                                          (bsxfun(@times,difc,dwd.') * model.psis{rightIdx}').' * difc;% - regularization(rightIdx)*f3
%             if LRprototypes > 0
% %                 G(k,:) = dcd * difw - dwd * difc; 
%             end            
        end
    end
    % some rescalings needed
    if LRrelevances > 0
%         G(nb_prototypes+1:nb_prototypes+n_vec,:) = 2/nb_samples * LRrelevances * Gw - regularization*f3;
        if sum(regularization)>0
            regmatrices = zeros(sum(nb_dims),nb_features);
            for i=1:length(model.psis)
                regmatrices(sum(nb_dims(1:i))-nb_dims(i)+1:sum(nb_dims(1:i)),:) = regularization(i) * (pinv(model.psis{i}))';
            end
            G(nb_prototypes+1:end,:) =  2/nb_samples * LRrelevances * cell2mat(Gw') - regmatrices;
%             G(nb_prototypes+1:end,:) =  cell2mat(Gw') - regmatrices;
        else
            G(nb_prototypes+1:end,:) =  2/nb_samples * LRrelevances * cell2mat(Gw');
%             G(nb_prototypes+1:end,:) =  cell2mat(Gw');
        end
    end
    if LRprototypes > 0
%         G(1:nb_prototypes,:) = 1./nb_samples * LRprototypes * G(1:nb_prototypes,:) * omegaT * omegaT.';
        G(1:nb_prototypes,:) = 1./nb_samples * LRprototypes * G(1:nb_prototypes,:);
    end
    G = G .* (1 + .0001 * (rand(size(G))-.5)); % help break symmetries

%     else    
%     G1 = zeros(size(variables)); % initially no gradient
%     dist =  zeros(nb_samples,length(model.c_w));
%     Js = zeros(1,nb_samples);
%     Ks = zeros(1,nb_samples);
%     dJs = zeros(1,nb_samples);
%     dKs = zeros(1,nb_samples);
%     DJs = zeros(nb_samples,nb_features);
%     DKs = zeros(nb_samples,nb_features);
%     normJforK = zeros(1,nb_samples);
%     normKforJ = zeros(1,nb_samples);
%     for i=1:nb_samples
%         % select one training sample randomly
%         xi = training_data(i,:);
%         c_xi = prototypeLabel(find(LabelEqualsPrototype(i,:)==1,1));
% %         dist = zeros(1,length(model.c_w));
%         if classwise,
%             for j=1:length(prototypeLabel)
%                 rightIdx = find(classes == model.c_w(j));
%                 dist(i,j) = (xi - model.w(j,:))*model.psis{rightIdx}'*(model.psis{rightIdx}*(xi - model.w(j,:))');
%             end
%         else
%             for j=1:length(prototypeLabel)
%                 dist(i,j) = (xi - model.w(j,:))*model.psis{j}'*(model.psis{j}*(xi - model.w(j,:))');
%             end
%         end
%         % determine the two winning prototypes
%         % nearest prototype with the same class
%         [sortDist,sortIdx] = sort(dist(i,:));
%         count = 1;
%         J = sortIdx(count);
%         while model.c_w(sortIdx(count)) ~= c_xi, 
%             count = count+1;
%             J = sortIdx(count);
%         end
%         dJ = sortDist(count);
%         count = 1;
%         K = sortIdx(count);
%         while model.c_w(sortIdx(count)) == c_xi, 
%             count = count+1;
%             K = sortIdx(count);
%         end
%         dK = sortDist(count);
% %         disp([J,K,dJ,dK]);
%         wJ = model.w(J,:);
%         wK = model.w(K,:);
%         % prototype update
%         norm_factor = (dJ + dK)^2;
%         DJ = (xi-wJ);
%         DK = (xi-wK);
%         DJs(i,:) = DJ;
%         DKs(i,:) = DK;
%         Js(i) = J;
%         Ks(i) = K;
%         dJs(i) = dJ;
%         dKs(i) = dK;
%         if classwise, 
%             rightIdxJ = find(model.c_w(J)==classes);
%             rightIdxK = find(model.c_w(K)==classes);
%         else
%             rightIdxJ = J;
%             rightIdxK = K;
%         end
%         normKforJ(i) = (2*dK/norm_factor)*2;
%         normJforK(i) = (2*dJ/norm_factor)*2;
%         dwJ = (2*dK/norm_factor)*2*model.psis{rightIdxJ}'*model.psis{rightIdxJ}*DJ';
%         dwK = (2*dJ/norm_factor)*2*model.psis{rightIdxK}'*model.psis{rightIdxK}*DK';
%         if LRprototypes > 0
% G1(J,:) = G1(J,:) - dwJ';
% G1(K,:) = G1(K,:) + dwK';
%         end
% %         model.w(J,:) = wJ + alphas(epoch) * dwJ';
% %         model.w(K,:) = wK - alphas(epoch) * dwK';
%         % update matrices
%         if LRrelevances > 0 % epoch >= MatrixStart
%             % compute updates for matrix omega
%             f1 =  (2*dK/norm_factor)*2*(model.psis{rightIdxJ}*DJ'*DJ);
%             f2 = (-2*dJ/norm_factor)*2*(model.psis{rightIdxK}*DK'*DK);
%             f3J = 0;
%             f3K = 0;
%             % update lambda & normalization
% %             for midx=1:length(nb_dims)
% %                 position = nb_prototypes+1+sum(nb_dims(1:midx-1));
% %                 position:position+nb_dims(midx)-1
% %             end
%             positionJ = nb_prototypes+1+sum(nb_dims(1:rightIdxJ-1));
%             positionK = nb_prototypes+1+sum(nb_dims(1:rightIdxK-1));
% G1(positionJ:positionJ+nb_dims(rightIdxJ)-1,:) = G1(positionJ:positionJ+nb_dims(rightIdxJ)-1,:) + (f1 - regularization(rightIdxJ) * f3J);
% G1(positionK:positionK+nb_dims(rightIdxK)-1,:) = G1(positionK:positionK+nb_dims(rightIdxK)-1,:) + (f2 - regularization(rightIdxK) * f3K);
% %             model.psis{rightIdxJ} = model.psis{rightIdxJ}-epsilons(epoch) * (f1 - regularization(rightIdxJ) * f3J);
% %             model.psis{rightIdxK} = model.psis{rightIdxK}-epsilons(epoch) * (f2 - regularization(rightIdxK) * f3K);
%         end
%     end
%     G = G1;
end
end