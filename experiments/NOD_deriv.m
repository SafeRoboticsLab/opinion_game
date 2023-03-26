clear all; close all; clc

z1 = sym('z1', [2,1]);
z2 = sym('z2', [2,1]);
V1 = sym('V1', [2,1]);
V2 = sym('V2', [2,1]);

z1e = [3; -3]; 
z2e = [3; -3];
V1e = [10; 100];
V2e = [100; 100];

Vhat1 = sigma(z1, 1)*sigma(z2, 1)*V1(1) +...
        sigma(z1, 1)*sigma(z2, 2)*V1(2) +...
        sigma(z1, 2)*sigma(z2, 1)*V2(1) +...
        sigma(z1, 2)*sigma(z2, 2)*V2(2);
  
dVhat1dz1 = gradient(Vhat1, z1);
H1 = - jacobian(dVhat1dz1, [z1; z2]);

% subs(H1, [z1;z2], ze)
pretty(H1(1,4))
pretty(H1(2,3))

H1s = double(subs(H1, [z1;z2;V1;V2], [z1e;z2e;V1e;V2e]))


%%
% clear all; close all; clc
% z1 = sym('z1', [3,1]);
% z2 = sym('z2', [3,1]);
% V1 = sym('V1', [3,1]);
% V2 = sym('V2', [3,1]);
% V3 = sym('V3', [3,1]);
% 
% z1e = [3; -3; -3]; 
% z2e = [3; -3; -3];
% V1e = [1; 100; 100];
% V2e = [100; 100; 100];
% V3e = [100; 100; 100];
% 
% Vhat1 = sigma(z1, 1)*sigma(z2, 1)*V1(1) +...
%         sigma(z1, 1)*sigma(z2, 2)*V1(2) +...
%         sigma(z1, 1)*sigma(z2, 3)*V1(3) +...
%         sigma(z1, 2)*sigma(z2, 1)*V2(1) +...
%         sigma(z1, 2)*sigma(z2, 2)*V2(2) +...
%         sigma(z1, 2)*sigma(z2, 3)*V2(3) +...
%         sigma(z1, 3)*sigma(z2, 1)*V3(1) +...
%         sigma(z1, 3)*sigma(z2, 2)*V3(2) +...
%         sigma(z1, 3)*sigma(z2, 3)*V3(3);
%   
% dVhat1dz1 = gradient(Vhat1, z1);
% H1 = - jacobian(dVhat1dz1, [z1; z2]);
% 
% double(subs(H1, [z1;z2;V1;V2;V3], [z1e;z2e;V1e;V2e;V3e]))
% 

%%
function sigma_zi = sigma(z, idx)
    sigma_zi = exp(z(idx))/sum(exp(z));
end

