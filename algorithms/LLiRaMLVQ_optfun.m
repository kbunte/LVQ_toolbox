function [f,G]  = LLiRaMLVQ_optfun(variables,trainSet,LabelEqualsPrototype,regularization,Lprototypes,Lrelevances,dim)
% [f G] = GMLVQ_optfun(variables) 
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

relIdx = find(isnan(variables(:,1))); % wIdx= 1:size(variables,1);wIdx(relIdx) = [];
wIdx = find(~isnan(variables(:,1)));
w   = variables(wIdx,2:end);
c_w = variables(wIdx,1);

omega = variables(relIdx(1:dim(2)),2:end);
psis = mat2cell(variables(relIdx(dim(2)+1:end),2:1+dim(2)),dim,dim(2))';
if length(c_w)~=length(dim), classwise = 1;else, classwise = 0;end %c_A = unique(round(c_w));classwise = 1;else c_A = c_w;classwise = 0;

P = size(trainSet,1);
dists = zeros(P,length(c_w));
if classwise
    classes = unique(c_w);
    for i=1:length(c_w)
        matrixIdx = classes==c_w(i);
        dists(1:P,i) = sum((bsxfun(@minus, trainSet, w(i,:))*omega'*psis{matrixIdx}').^2, 2);
    end
else
    for i=1:length(c_w)
        dists(1:P,i) = sum((bsxfun(@minus, trainSet, w(i,:))*omega'*psis{i}').^2, 2);
    end
end
Dwrong = dists;
Dwrong(LabelEqualsPrototype) = realmax(class(Dwrong));   % set correct labels impossible
[distwrong, pidxwrong] = min(Dwrong.'); % closest wrong
clear Dwrong;

Dcorrect = dists;
Dcorrect(~LabelEqualsPrototype) = realmax(class(Dcorrect)); % set wrong labels impossible
[distcorrect, pidxcorrect] = min(Dcorrect.'); % closest correct
clear Dcorrect;

distcorrectpluswrong  = distcorrect + distwrong;
distcorrectminuswrong = distcorrect - distwrong;

mu = distcorrectminuswrong ./ distcorrectpluswrong;
regTerm = 0;
if regularization
    error('regularization not implemented yet!');
%     regTerm = regularization * log(det(A*A'));
    for j=1:length(Aidx)
        regTerm = regTerm + regularization * log(det(A(Aidx{j},:)*A(Aidx{j},:)'));
    end
end
normTerm = 0;% sum(diag(omega'*omega))
normTerm = normTerm + (1-sum(arrayfun(@(d) sum(omega(:,d).*omega(:,d)),1:size(variables,2)-1)))^2 + sum((1-cellfun(@(x) sum(diag(x'*x)),psis)).^2);
% cellfun(@(x) sum(diag(x'*x)),psis)
% for j=1:length(Aidx)
%     normTerm = normTerm + (1-sum(arrayfun(@(d) sum(A(d,:).*A(d,:)),Aidx{j})))^2;
% end
% f = sum(mu) -regTerm + normTerm;
% if one uses mean here to have the normalizing terms more enforced one needs to divide the gradients by n as well!
f = mean(mu) -regTerm + normTerm;
% f = sum(mu); 

if nargout > 1  % gradient needed not just function eval
    G = zeros(size(variables)); % initially no gradient
    dO = zeros(size(omega));
    dP = cellfun(@(x) zeros(size(x)),psis,'uni',0);
        
    nb_prototypes = length(c_w);
%     mudJ =  2*distwrong  ./(distcorrectpluswrong.^2);
%     mudK = -2*distcorrect./(distcorrectpluswrong.^2);
    normfactors = 4 ./ distcorrectpluswrong.^2;
    for k=1:nb_prototypes
        idxc = (k == pidxcorrect);  % Js: idxs where actual prototype is nearest correct
        idxw = (k == pidxwrong);    % Ks: idxs where actual prototype is nearest wrong            
        if classwise, matrixIdx = classes==c_w(k);else, matrixIdx = k;end % TODO: classwise not tested!

        dcd =  distcorrect(idxw) .* normfactors(idxw); % 4*dJ/(dJ+dK)^2 where actual w is nearest wrong
        dwd =    distwrong(idxc) .* normfactors(idxc); % 4*dK/(dJ+dK)^2 where actual w is nearest correct
        % part of derivative of distance
        difc = bsxfun(@minus,trainSet(idxc,:),w(k,:)); % DJs
        difw = bsxfun(@minus,trainSet(idxw,:),w(k,:)); % DKs
        % prototype update
        if Lprototypes
            G(k,2:end) = 1/P.*(dcd*difw - dwd*difc)*omega'*psis{matrixIdx}'*psis{matrixIdx}*omega; % if f=mean(mu) is used do not forget to divide by n
        end
        % relevance updates
        if Lrelevances            
            h1 = bsxfun(@times,difc,dwd')'*difc;
            h2 = bsxfun(@times,difw,dcd')'*difw;
            dO = dO + psis{matrixIdx}'*psis{matrixIdx}*omega *2* (h1 - h2);
            dP{matrixIdx} = dP{matrixIdx} + 2*psis{matrixIdx}*(omega*(h1 - h2))*omega';
%             tic;
%                 test = (dwd.*difc')';
%             toc
%             tic;
%                 test2= bsxfun(@times,difc,dwd');
%             toc
%             f1 = bsxfun(@times,difc,dwd')'*difc;
%             f2 = bsxfun(@times,difw,dcd')'*difw;
%             dO = dO + psis{matrixIdx}'*psis{matrixIdx}*omega *2* (f1 - f2);
%             f1 = (dwd.*difc')*difc;
%             f2 = (dcd.*difw')*difw;
%             (mudJ(idxc).*difc')*difc - (dwd.*difc')*difc
%             (mudK(idxw).*difw')*difw + (dcd.*difw')*difw
%             dP{matrixIdx} = dP{matrixIdx} + 2*psis{matrixIdx}*(omega*(f1 - f2))*omega';
%             [mudJ(idxc)',dwd']
%             [mudK(idxw)',dcd']            
%             f2 = psis{matrixIdx}'*psis{matrixIdx}*omega * (2.*mudJ(idxc)'.*difc)'*difc;
%             f1 = psis{matrixIdx}'*psis{matrixIdx}*omega * (2.*mudK(idxw)'.*difw)'*difw;
%             dO = dO + (f1+f2);
% 
%             f1 = 2*psis{matrixIdx}*(omega*(mudJ(idxc).*difc')*difc)*omega';
%             f2 = 2*psis{matrixIdx}*(omega*(mudK(idxw).*difw')*difw)*omega';
%             dP{matrixIdx} = dP{matrixIdx} + (f1+f2);
% 
%             f2 = psis{matrixIdx}'*psis{matrixIdx}*omega * (2.*mudJ(idxc)'.*DJ)'*DJ;
%             f1 = psis{matrixIdx}'*psis{matrixIdx}*omega * (2.*mudK(idxw)'.*DK)'*DK;
%             dO = dO + (f1+f2);            
%             f1 = 2*psis{matrixIdx}*(omega*(mudJ(idxc).*DJ')*DJ)*omega';
%             f2 = 2*psis{matrixIdx}*(omega*(mudK(idxw).*DK')*DK)*omega';
%             dP{matrixIdx} = dP{matrixIdx} + (f1+f2);
        end
    end
    if Lrelevances
        % TODO: regularization not implemented!
        normOterm = -4*(1-sum(arrayfun(@(d) sum(omega(:,d).*omega(:,d)),1:size(omega,2)))).*omega;
        normPterms= cellfun(@(x) -4*(1-sum(diag(x'*x))).*x ,psis,'uni',0);            
% omega = variables(relIdx(1:dim(2)),2:end);
% psis = mat2cell(variables(relIdx(dim(2)+1:end),2:1+dim(2)),dim,dim(2))';  

        G(relIdx(1:dim(2)),2:end) = 1/P.*dO + normOterm;            
        gpsis = cell2mat(arrayfun(@(j) 1/P.*dP{j} + normPterms{j},1:length(dP),'uni',0)');            
        G(relIdx(dim(2)+1:end),2:1+dim(2)) = gpsis;
%         this is without normalization terms:
%         G(relIdx(1:dim(2)),2:end) = 1/P.*dO;
%         gpsis = cell2mat(arrayfun(@(j) dP{j},1:length(dP),'uni',0)');
%         G(relIdx(dim(2)+1:end),2:1+dim(2)) = 1/P.*gpsis;
    end
end
end
