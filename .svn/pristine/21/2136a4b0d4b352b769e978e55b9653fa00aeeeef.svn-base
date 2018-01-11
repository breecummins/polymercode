function [xold,yold] = RegriddingProcedure

% program to compute regridding 
%


% make fake data at random points
h_old = 0.1;
[Xold,Yold]=meshgrid( 0 : h_old : 2, 0 : h_old : 2);

th = 0.233;
A = [[cos(th) -sin(th)];[sin(th) cos(th)]];
xold = cos(th)*Xold - sin(th)*Yold;
yold = sin(th)*Xold + cos(th)*Yold;

%the function will be
Fold = exp(-((xold-1).^2+(yold-1).^2)*6);
Fold = sin(xold).*cos(yold)+1;
Fold = ones(size(xold));


%place new points on a grid of size h
h0 = 0.1;  pad = 4;
xmin = min(min(xold))-pad*h0;
xmax = max(max(xold))+pad*h0;
ymin = min(min(yold))-pad*h0;
ymax = max(max(yold))+pad*h0;

Nx = fix((xmax-xmin)/h0); h = (xmax-xmin)/Nx;
Ny = fix((ymax-ymin)/h); 
[xnew,ynew] = meshgrid( xmin : h : xmax, ymin : h : ymax );
xp = (xmin : h : xmax)';  yp = (ymin : h : ymax)';


% CHOOSE AN INTERPOLANT: 2, 4 OR 6
whichinterp = 4;

%parameters
%====================
delta = 3*h;
dist = round(delta/h);

%at every grid point, there will be a coefficient for each vector
Fnew = zeros(size(xnew));  size(Fnew), pause(1)
%end of parameters
%%%%%%%%
%%



%====================
disp(' ... going into regridding loop')
Npt = length(xold(:));  x_old = xold(:); y_old = yold(:); F_old = Fold(:);
% for each point, go through the 8 vectors
for jpt = 1 : Npt
    
    % find the patch of grid where this point will interpolate
    i0 = floor((x_old(jpt)-xmin)/h)+1;     j0 = floor((y_old(jpt)-ymin)/h)+1;
	id1 = i0 + (1-dist:dist);     id2 = j0+ (1-dist:dist);

    rx = xp(id1)-x_old(jpt);   	ry = yp(id2)-y_old(jpt); 
    r1 = rx/h;                  r2 = ry/h;
    
    %get the patch of weights
    gpatch = blobproduct(1,r1,r2,delta,h,whichinterp); 
    Fnew(id1,id2) = Fnew(id1,id2) + F_old(jpt)*gpatch*h_old*h_old/(h*h);


end

disp(' ... out of regridding loop')

%%

%for some reason this ends up reversing the x and y axes
figure(1),mesh(ynew,xnew,Fnew)
figure(2),mesh(xold,yold,Fold)


keyboard

    
end%function
%%%%%%%%%%%%%%%
	  
function [for1] = blobproduct(f1,rx,ry,delta,h,which)
%
% THIRD ORDER FUNCTION IN [-3,3]
%
scale = 3*h/delta;
arx = abs(rx);   ary = abs(ry);

%  We need to make everything a column and scale
r = scale*rx(:);      ar = scale*arx(:); 

%  Make space for blob parts
deltaX = zeros(size(r));   
deltaY = deltaX;          

wd=doit(r,ar,which); 

for k = 1 : 3
 deltaX = deltaX + wd(:,k) .* (abs(r)>=k-1) .* (abs(r)<k);
end
deltaX = deltaX*scale;   


%  Now do it all again in y
r = scale*ry(:);     ar = scale*ary(:);

wd=doit(r,ar,which);

for k = 1 : 3
  deltaY = deltaY + wd(:,k) .* (abs(r)>=k-1) .* (abs(r)<k);
end
deltaY = deltaY*scale;  


for1= f1*(deltaX)*(deltaY');


end%blobproduct
%--------------------------------------------
function wd = doit(r,ar,which)
%--------------------------------------------

wd = zeros(length(r), 3); 

switch which

 case 2, % SECOND ORDER C^1 DELTA FUNCTION WITH SUPPORT [-3,3]
  
  wd(:,1) = 5/8 - (3/8)*r.^2;
  wd(:,2) = 23/16  - 13/8*ar + (7/16)*r.^2;
  wd(:,3) = -1/16*(ar-3).^2;
  
 case 4, % THIRD ORDER DELTA FUNCTION WITH SUPPORT [-2,2]
  
  wd(:,1) = 1 - 5/2*r.^2 + 3/2*ar.^3; 
  wd(:,2) = 1/2*(2-ar).^2 .* (1-ar);
  wd(:,3) = zeros(size(ar));
  
 case 6, % FOURTH ORDER DELTA FUNCTION WITH SUPPORT [-3,3]
  
  wd(:,1) = -1/12*(ar-1).*(25*r.^4 - 38*ar.^3 - 3*r.^2 + 12*ar + 12); 
  wd(:,2) = 1/24*(ar-1).*(ar-2).*(25*ar.^3 - 114*r.^2 + 153*ar - 48);
  wd(:,3) = -1/24*(ar-2).*(ar-3).^3.*(5*ar-8);

 otherwise,
  display('bad switch in blobproduct, doit',which)
  pause
end
end%doit

