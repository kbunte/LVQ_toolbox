function mappedX=manch(no_analyzers,no_dims,X,R,Z)

% Adds last entry = 1 in posterior mean to handle means of factor analyzers
Z(no_dims + 1,:,:) = 1; 
Z = permute(Z, [1 3 2]);

  kf = no_analyzers * (no_dims + 1);
    % Construct blockdiagonal matrix D
    disp('Performing manifold charting...');
    D = zeros((no_dims + 1) * no_analyzers, (no_dims + 1) * no_analyzers);
    for i=1:no_analyzers
        Ds = zeros(no_dims + 1, no_dims + 1);
        for j=1:size(X, 1)
            Ds = Ds + R(i, j) .* (Z(:,i,j) * Z(:,i,j)');
        end
        D((i - 1) * (no_dims + 1) + 1:i * (no_dims + 1), (i - 1) * (no_dims + 1) + 1:i * (no_dims + 1)) = Ds;
    end
    
    % Construct responsibility weighted local representation matrix U
    R = reshape(R, [1 no_analyzers size(X, 1)]);
    U = reshape(repmat(R, [no_dims + 1 1 1]) .* Z, [kf size(X, 1)])';

    % Solve generalized eigenproblem
        options.disp = 0;
        options.isreal = 1;
%         options.issym = 1;

        [V, lambda] = eigs(D - U' * U, U' * U, no_dims + 1, 'SM', options);% 

    [lambda, ind] = sort(diag(lambda));
    V = V(:,ind(2:end));
    
    % Compute final lowdimensional data representation
    mappedX = U * V;
    