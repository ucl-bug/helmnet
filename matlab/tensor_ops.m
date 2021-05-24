function result = tensor_ops(M, operation, varargin)
    M = sparse(M);
    switch operation
        case 'matrix_to_vector'
            result = M(:);

        case 'matrix_to_elementwise'
            result = diag(M(:));

        case 'matrix_to_LHS_product'
            result = kron(speye(size(M)),M); 

        case 'matrix_to_RHS_product'
            result = kron(transpose(M),speye(size(M)));

        case 'vector_to_matrix'
            matrix_shape = varargin{1};
            result = reshape(M, matrix_shape(1),matrix_shape(2));

        otherwise
            error('Unknown operation')
    end
end
    