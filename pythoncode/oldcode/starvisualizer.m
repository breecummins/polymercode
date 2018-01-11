%springvisualizer.m

clear
close all

set(0,'DefaultAxesFontSize',24)

%%%%%%%%%%%%%%%%%%%%%%  Two head spring %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vefile = '~/scratch/star2_visco.mat';
load(vefile)

%%%%%%%%%%%%%%%%%%
set(0,'DefaultAxesFontSize',24)

figure
set(gcf, 'PaperSize', [8 6]);
set(gcf, 'PaperPosition', [0,0,8,6]);
set(gcf, 'Color','k')
Xmasmap=[linspace(0,0.95,64).',linspace(0.75,0,64).',zeros(64,1)];
for k = length(t)-10;%1:length(t)  
	clf
	
	%make the star and blur it
	%first build a dense array around the object
	startpt = min(fpts(k,:))-0.025;
	endpt = max(fpts(k,:))+0.025;
	x = linspace(startpt,endpt,150);
	[X,Y] = meshgrid(x,x);
	locmask = zeros(size(X));
	for gp = 1:size(fpts,2)/2;
		if gp == size(fpts,2)/2;
			linex = linspace(fpts(k,2*gp-1),fpts(k,1),50);
			liney = linspace(fpts(k,2*gp),fpts(k,2),50);
		else;	
			linex = linspace(fpts(k,2*gp-1),fpts(k,2*gp+1),50);
			liney = linspace(fpts(k,2*gp),fpts(k,2*gp+2),50);
		end
		for lp = 1:length(linex);
			xdiff = linex(lp) - X;
			ydiff = liney(lp) - Y;
			ind = find(abs(xdiff) == min(min(abs(xdiff))));
			jnd = find(abs(ydiff) == min(min(abs(ydiff))));
			match = intersect(ind,jnd);
			if ~isempty(match) && length(match) == 1;
				locmask(match) = 1;  %fill in the grid points that are closest to the original object
			else;
				disp('Problem with grid construction. Match = ')
				disp(match)
				error()
			end
		end
	end
	%now build a function to convolve with the mask
	h = x(2)-x(1);
	s= 2*h; %support of spread function
	xg = -s:h/2:s; 
	C = 4*h^2; %gauss 2*std dev^2
	xg = -5*h:h/2:5*h; 
	[Xg,Yg] = meshgrid(xg,xg);
	%wignersemicirc = (1/s)*sqrt(abs(s^2-Xg.^2-Yg.^2));
	gauss = exp(-(Xg.^2+Yg.^2)/C);
	fuzzyimage = conv2(locmask,gauss,'same');
	% %the convolution spreads things too much -- apply a threshold
	fuzzyimage = 2*fuzzyimage/max(max(fuzzyimage));
	mask1 = (fuzzyimage > 1);
	fuzzyimage(mask1) = 1;
	% mask = (fuzzyimage > 0.25);
	% fuzzyimage(mask) = 1;
	
	%get the stress and find the clims
	lx = squeeze(l(k,:,:,1));
	ly = squeeze(l(k,:,:,2));
	Str = squeeze(Strace(k,:,:));
	surf(lx,ly,Str,'FaceColor','interp','EdgeColor','none','FaceLighting','phong')
	clims = caxis;
	
	%plot the star first, then the stress trace, then apply clims (only way I colud this to work -- gray magic)
	surf(X,Y,3*ones(size(X)),'FaceAlpha','interp','AlphaDataMapping','scaled','AlphaData',fuzzyimage,'FaceColor',[1,1,0],'EdgeColor','none')
	hold on
	surf(lx,ly,Str,'FaceColor','interp','EdgeColor','none','FaceLighting','phong')
	axis equal
	axis off
	colormap(Xmasmap)
	caxis(clims);
	view(2)	
	
	hold off
	
	pause(2.0)
	% if mod(k,10) == 1;	
	% 	saveas(gcf,['~/scratch/star',sprintf('%03d',c),'.png'],'png')
	% else;
	% 	pause(0.001)
	% end
end

