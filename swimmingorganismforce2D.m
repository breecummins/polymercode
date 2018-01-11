function h = swimmingorganismforce2D

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%input all parameters here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%ORGANISM WILL BE:  
%  xt = s;  yt = a*s*sin(lambda*s-W*t);
%
Np = 20;  %number of points in the organism
a = 0.1;  %maximum amplitude at the tail (zero at the head). Approx.
W = 2*pi; %frequency
L = 0.75; %period for "s"
mwv = 1.25; %number of periods in the organism
lambda = 2*pi*mwv/L; %wave number

[h,xf]=newtonFish(a,Np); %find equally spaced points as Initial Cond.
del = h/2; %blob size

%find spring information (spring k connects points k and k+1)
sprg = zeros(Np-1,4);
for k=1:Np-1
    sprg(k,:) = [k, k+1, h, 40];
end
StiffCurv = 0.01;

%start time loop
Tf = 1;
dt = 0.0001;

for t = 0 : dt : Tf
    
   %find the forces
   Force = getForces(t,xf,h,sprg,[a,W,lambda,StiffCurv]);
   %compute the velocity
   U = GetVelocity(xf,xf,Force,del);
   %update the points
   xf = xf + dt * U;

   %temporary plotting
   if mod(t,100*dt) == 0;
    quiver(xf(:,1),xf(:,2),Force(:,1),Force(:,2)),hold on,
    plot(xf(:,1),xf(:,2),'r.',xf(:,1),xf(:,2),'r')
    axis equal,axis([-0.1 0.9 -0.2 0.2])
    grid 
    title(['time = ',num2str(t)])
   end

   pause(.1),hold off

end %time

keyboard

end%swimmingorganismforce2D

%%%%%%%%%%%OOOOOOOOOOOOOOOOOOOOO%%%%%%%%%%%
%%%%%%%%%%%OOOOOOOOOOOOOOOOOOOOO%%%%%%%%%%%
function Force = getForces(t,pts,h,sprg,CurvParams)
%
% pts = (Np,2) array containing the coordinates of Np points
% h = 'delta s' or point separation (currently constant)
% sprg = (Ns,4) array containing information about Ns springs
%      = [p1,p2,rest_length,stiffness]
%      
     
Np = length(pts(:,1));
Ns = length(sprg(:,1));

% split the arrays for simplicity
xf = pts(:,1);   yf = pts(:,2);


% forces will be two types: F1 = springs, F2 = curvature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SPRING FORCES: F1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F1 = zeros(Np,2);
for k = 1 : Ns
    p1   = sprg(k,1);
    p2   = sprg(k,2);
    rlen = sprg(k,3);
    stff = sprg(k,4);
    
    dx  = xf(p1)-xf(p2);
    dy  = yf(p1)-yf(p2);
    sep = sqrt( dx^2 + dy^2 );
    
    tmp1 = stff*(sep/rlen - 1) * dx./rlen;
    tmp2 = stff*(sep/rlen - 1) * dy./rlen;
    
    F1(p2,1) = F1(p2,1) + tmp1;
    F1(p2,2) = F1(p2,2) + tmp2;
    
    F1(p1,1) = F1(p1,1) - tmp1;
    F1(p1,2) = F1(p1,2) - tmp2;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CURVATURE FORCES: F2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F2 = zeros(Np,2);

% We have to compute the desired curvature first: curv=-y''
% assuming equally spaced points in arclength

a = CurvParams(1);     
W = CurvParams(2);    
lambda = CurvParams(3);
StiffC = CurvParams(4);

xt = (0 : h : (Np-1)*h)';
yt = a*xt.*sin(lambda*xt-W*t);
curv = a*lambda^2*xt.*sin(lambda*xt-W*t) - 2*a*lambda*cos(lambda*xt-W*t);

% First: forces on points that have at least two neighbors on either side
% We assume the points are in order from head to tail

for n = 2:1:Np-1
    
    dxkp1 = (xf(n+1)-xf(n)  )/h;  dykp1 = (yf(n+1)-yf(n)  )/h;
    dxk   = (xf(n)  -xf(n-1))/h;  dyk   = (yf(n)  -yf(n-1))/h;
    
    CRSSk   = (dxkp1*dyk - dxk*dykp1)/h;
     
    F2(n+1,1) = F2(n+1,1) + StiffC*( CRSSk-curv(n) ) .* dyk/h;
    F2(n+1,2) = F2(n+1,2) - StiffC*( CRSSk-curv(n) ) .* dxk/h;
    
    F2(n,1) = F2(n,1) - StiffC*( CRSSk-curv(n) ) .* (dyk+dykp1)/h;
    F2(n,2) = F2(n,2) + StiffC*( CRSSk-curv(n) ) .* (dxk+dxkp1)/h;
    
    F2(n-1,1) = F2(n-1,1) + StiffC*( CRSSk-curv(n) ) .* dykp1/h;
    F2(n-1,2) = F2(n-1,2) - StiffC*( CRSSk-curv(n) ) .* dxkp1/h;

end

Force = F1-F2;

end%getForces
%%%%%%%%%%%OOOOOOOOOOOOOOOOOOOOO%%%%%%%%%%%
%%%%%%%%%%%OOOOOOOOOOOOOOOOOOOOO%%%%%%%%%%%
function U = GetVelocity(Xeval,Xforce,F,del)

NTot = length(Xeval(:,1));
del2 = del*del;

xf = Xforce(:,1);
yf = Xforce(:,2);

x = Xeval(:,1);  u = zeros(size(x));
y = Xeval(:,2);  v = zeros(size(y));

f1 = F(:,1);
f2 = F(:,2);

for k=1:NTot
    
   dx = x(k)-xf;
   dy = y(k)-yf;
   r2 = dx.^2+dy.^2;
   
   H1 = (2*del2)./(r2+del2) - log(r2+del2);
   H2 = 2./(r2+del2);
   fdotx = dx.*f1 + dy.*f2; 
   
   u(k) = sum(H1.*f1 + H2.*fdotx.*dx);
   v(k) = sum(H1.*f2 + H2.*fdotx.*dy);

end
U = [u,v]/(8*pi);
end %function
%%%%%%%%%%%OOOOOOOOOOOOOOOOOOOOO%%%%%%%%%%%
%%%%%%%%%%%OOOOOOOOOOOOOOOOOOOOO%%%%%%%%%%%
function [mds,x]=newtonFish(a,ntot)
% x = newt2(a,n) RETURNS n POINTS EQUALLY SPACED
% (IN EUCLIDEAN DISTANCE) ALONG THE CURVE
%   y = a * t * sin(lambda*t)
% THE OUTPUT IS THE n-by-2 ARRAY x = [t y]
%
% EXAMPLE:
%          x = newt2(0.1,20);
%
% USE THE FUNCTION
% [x,y] = getxy(a,t);
%

% PARAMETERS

Ly = 0.75;   m = 1.25;%( m = number of sine waves in x )
lambda = 2*pi*m/Ly;

n = ntot-2;

% INITIAL GUESS FOR x
t0 = 0;  tnp1 = Ly;
dt = (tnp1-t0)/(n+1);
t = (t0+dt:dt:tnp1-dt);

[x0,f0]     = getxy(a,lambda,t0);
[xnp1,fnp1] = getxy(a,lambda,tnp1);


% BEGIN THE ITERATION UP TO
nlast = 20;
tol   = 1e-7;
Fmax = 1; iter = 0;

%for i=1:nlast,
while Fmax > tol && iter < nlast,

% FIRST FIND  x(t),f(t)
[x,f] = getxy(a,lambda,t);

% COMPUTE THE VECTOR FUNCTION
xkp1 = [x(2:n) xnp1];   fkp1 = [f(2:n) fnp1];
xkm1 = [x0 x(1:n-1)];   fkm1 = [f0 f(1:n-1)];

F = (x-xkm1).^2 + (f-fkm1).^2 - (xkp1-x).^2 - (fkp1-f).^2;
Fmax = max(abs(F));

% NOW FIND x'(t),f'(t)
[dx,df] = getdxdy(a,lambda,t);

A = zeros(n,n);
% SET UP THE MATRIX
A(1,1) = 2*(x(1)-x0)*dx(1) + 2*(f(1)-f0)*df(1) ...
       + 2*(x(2)-x(1))*dx(1) + 2*(f(2)-f(1))*df(1); 
A(1,2) = -2*(x(2)-x(1))*dx(2) - 2*(f(2)-f(1))*df(2); 
for k=2:n-1,
  A(k,k-1) = -2*(x(k)-x(k-1))*dx(k-1) - 2*(f(k)-f(k-1))*df(k-1); 
  A(k,k)   = 2*(x(k)-x(k-1))*dx(k) + 2*(f(k)-f(k-1))*df(k) ...
           + 2*(x(k+1)-x(k))*dx(k) + 2*(f(k+1)-f(k))*df(k); 
  A(k,k+1) = -2*(x(k+1)-x(k))*dx(k+1) - 2*(f(k+1)-f(k))*df(k+1); 
end
A(n,n-1) = -2*(x(n)-x(n-1))*dx(n-1) - 2*(f(n)-f(n-1))*df(n-1); 
A(n,n)   = 2*(x(n)-x(n-1))*dx(n) + 2*(f(n)-f(n-1))*df(n) ...
         + 2*(xnp1-x(n))*dx(n) + 2*(fnp1-f(n))*df(n); 

% INVERT THE MATRIX
Ain = inv(A);

% NEW VALUES
v = t - F*Ain';
t = v;

iter = iter + 1;
end

t = [t0 t tnp1]';
[xx,ff] = getxy(a,lambda,t);

plot(xx,ff,'b',xx,ff,'r.')
axis equal
xt=[xx];
yt=[ff];
x=[xt yt];

dx = diff(x(:,1));
dy = diff(x(:,2));
ds = sqrt( dx.^2 + dy.^2 );  mds = mean(ds);

end%newt2

%+++++++++++++++++++++++++++++
function [x,y]=getxy(a,lambda,t)
   
   x = t;
   y = a*t.*sin(lambda*t);
end%getxy
%+++++++++++++++++++++++++++++
% these must be the derivatives of the ones above
function [dx,dy]=getdxdy(a,lambda,t)
      
   dx = ones(size(t));
   dy = a*sin(lambda*t) + a*lambda*t.*cos(lambda*t);
end%getdxdy
%%%%%%%%%%%OOOOOOOOOOOOOOOOOOOOO%%%%%%%%%%%



