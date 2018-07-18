function [d, loss] = del_beta(beta, x, y)

% function to calculate delta_beta for Gauss-Newton optimization
% It is a closed-form function for sinusoidal regression function
% d: delta beta
% loss: loss function value
% J: Jacobian Matrix
% r: Residual Vector

    j11 = sin(beta(2) .* x + beta(3));
    j12 = beta(1) .* x .* cos(beta(2) .* x + beta(3));
    j13 = beta(1) .* cos(beta(2) .* x + beta(3));
    j14 = ones(1,numel(x));
    J = [j11; j12; j13; j14]';
    f = beta(1) * sin(beta(2) .* x + beta(3)) + beta(4);
    r = (y - f)';
    d = (J'*J)\J'*r;
    loss = norm(r,2);
end