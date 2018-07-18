% Gauss-Newton Method for SinveWave Regression

clear all
clc
rng(991)
threshold = 1e-6;
alpha = 0.01;

A = 0.5;
B = 1.2;
C = 0.3;
D = 0.6;
noise_lvl = -10;
x = 0:0.01:10;
y = A * sin(B .* x + C) + D + wgn(1,numel(x),noise_lvl);

beta = rand(1,4);

loss = [];
diff = 1;
i = 0;
temp = 1;

while diff > threshold
    i = i + 1;
    [del, temp] = del_beta(beta, x, y);
    beta = beta + alpha * del';
    loss = [loss temp];
    if i == 1
        diff = abs(loss);
    else
        diff = abs(loss(end) - loss(end-1));
    end
end

disp(['Presumed Paramaters [A B C D]:  ', num2str([A B C D])]);
disp(['Estimated Paramaters [A B C D]: ', num2str(beta)]);
plot(loss);title('Loss function');xlabel('#iteration')

figure
plot(x,y,'.'); hold on; 
plot(x,beta(1) * sin(beta(2) .* x + beta(3)) + beta(4),'LineWidth',1);
legend('original data','regression line')

LOSS = zeros(100,100);
for i = 1:100
    for j = 1:100
        f = A * sin((i/10) .* x + (j*2*pi/100)) + D;
        r = (y - f)';
        LOSS(i,j) = norm(r,2);
    end
end

figure
s = surf([1:100]*2*pi/100,[1:100]/10,LOSS,'FaceAlpha',0.5);
s.EdgeColor = 'none';
xlabel('phase angle');ylabel('frequency')
zlabel('function value')
