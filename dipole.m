% Dipole




h = 1/20; [xg,yg]=meshgrid(-1:h:1);


Nh = 1;
theta = rand(Nh,1)*2*pi;
f = 2*pi*d^2*[cos(theta) sin(theta)];
xf = 0.25*(rand(Nh,2)-1/2);

%find velocity
d = 0.25;
ug = zeros(size(xg));  vg = ug;
for k = 1:Nh
    dx = xg - xf(k,1);
    dy = yg - xf(k,2);
    r2 = dx.^2 + dy.^2;
    
    D1 = (d^2-r2)./(d^2+r2).^2;
    D2 = 2./(d^2+r2).^2;
    
    fdotx = f(k,1)*dx + f(k,2)*dy;
    
    ug = ug + f(k,1)*D1 + fdotx.*dx.*D2;
    vg = vg + f(k,2)*D1 + fdotx.*dy.*D2;
    
end
ug = ug/(2*pi);  vg = vg/(2*pi);




quiver(xf(:,1),xf(:,2),f(:,1),f(:,2),'r'),hold on
quiver(xg,yg,ug,vg,2),axis equal
hold off