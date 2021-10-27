clear
close all
syms x y
t = 10^-2;
K = 1000;
k = 1;
x_init = 5;
x_k0 = x_init;
y_init = -9;
y_k0 = y_init;
x_k1 = x_init - t * grad_f_x(x_init,y_init);
y_k1 = y_init - t * grad_f_y(x_init,y_init);
iters = zeros(K-1,2);
gamma_k1 = 1;
Beta = zeros(K-1,1);
p = 0;
while (k < K)
gamma_k2 = 1/2 * (4 * gamma_k1^2 + gamma_k1^4)^(1/2) - gamma_k1^2;
if k< floor(t*K)
beta = -gamma_k2 * (1 - 1/gamma_k1);
g_k2 = x_k1 + beta*(x_k1 - x_k0);
j_k2 = y_k1 + beta*(y_k1 - y_k0);
x_k2 = g_k2 - t * grad_f_x(g_k2,j_k2);
y_k2 = j_k2 - t * grad_f_y(g_k2,j_k2);
else
beta = (p*norm((x_k1 - x_k0)).^2)';
j_k2 = y_k1 + beta*(y_k1 - y_k0);
g_k2 = x_k1 + beta*(x_k1 - x_k0);
y_k2 = j_k2 - t * grad_f_y(g_k2,j_k2);
x_k2 = g_k2 - t * grad_f_x(g_k2,j_k2);
end
gamma_k1 = gamma_k2;
iters(k,1)= x_k0 ;
iters(k,2)= y_k0 ;
Beta(k)= beta;
k =k + 1;
x_k0 = x_k1;
x_k1 = x_k2;
y_k0 = y_k1;
y_k1 = y_k2;
p = t*k-1;
end
f = x^2+10*y^2;
t = 10^-2;
K = 1000;
x_init1 = 5;
y_init1 = -9;
H = GD(x_init,y_init,K,t);
[A, B] = WGD(x_init1,y_init1,K,t);
title('Initial condition: (x_0,y_0)')
colorbar
fcontour(f,'fill','off','MeshDensity',500)
grid on
xlabel('x')
ylabel('y')
hold on
x_n = iters(:,1);
y_n = iters(:,2);
x_w = A(:,1);
y_w = A(:,2);
x_g = H(:,1);
y_g = H(:,2);
plot(x_n,y_n,'red')
plot(x_w,y_w,'blue')
plot(x_g,y_g,'black')
hold off
function H = GD(x_init,y_init,K,t)
k = 1;
H = zeros(K-1,2);
while (k < K)
x_k = x_init - t * grad_f_x(x_init,y_init);
H(k,1)= x_k ;
x_init = x_k;
y_k = y_init - t * grad_f_y(x_init,y_init);
H(k,2)= y_k ;
y_init = y_k ;
k = k+1;
end
end
function [A,B] = WGD(x_init,y_init,K,t)
k = 1;
x_k0 = x_init;
y_k0 = y_init;
x_k1 = x_init - t * grad_f_x(x_init,y_init);
y_k1 = y_init - t * grad_f_y(x_init,y_init);
A = zeros(K-1,2);
B = zeros(K-1,1);
while (k < K)
beta_x = sqrt(t)+ k*t*((x_k1 - x_k0)'*(x_k1 - x_k0));
x_k2 = x_k1 +(1- beta_x)*(x_k1 - x_k0) - t*grad_f_x(x_k1,y_k1);
B(k) = (x_k1 - x_k0);
A(k,1)= x_k2 ;
k =k + 1;
x_k0 = x_k1;
x_k1 = x_k2;
beta_y = sqrt(t)+ k*t*((y_k1 - y_k0)'*(y_k1 - y_k0));
y_k2 = y_k1 +(1- beta_y)*(y_k1 - y_k0) - t*grad_f_y(x_k1,y_k1);
A(k,2)= y_k2 ;
y_k0 = y_k1;
y_k1 = y_k2;
end
end
function g1 = grad_f_x(x,y)
 g1 = 2*x;          
end
function g2 = grad_f_y(x,y)
 g2 = 20*y;
end