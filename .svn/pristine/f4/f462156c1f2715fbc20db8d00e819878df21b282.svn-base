%star_tree_visualizer.m

clear
close all

set(0,'DefaultAxesFontSize',24)

%%%%%%%%%%%%%%%%%%%%%%  Two head spring %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vefile = '~/scratch/star_tree2_visco.mat';
load(vefile)
n = pdict.n;
mpts = fpts(:,2*n+1:end);
fpts = fpts(:,1:2*n);

%%%%%%%%%%%%%%%%%%
set(0,'DefaultAxesFontSize',24)
%bgcolor = [0,0,1];
bgcolor = 'w';
%bgcolor = [0,0,0.5];
Xmasmap=[linspace(0,1,64).',linspace(1,0,64).',zeros(64,1)];
frames = [2,24,length(t)-5];
%frames = length(t)-5;
% cmin = min(min(min(Strace(frames,:,:))));
% cmax = max(max(max(Strace(frames,:,:))));
% df = cmax - cmin;
% cmax = cmax - 0.25*df;

c=0;
for k = frames;
	c=c+1;
	figure
	set(gcf, 'PaperSize', [11/3, 8.5]);
	set(gcf, 'PaperPosition', [0,0,11/3, 8.5]);
	set(gcf, 'Color',bgcolor)
	set(gcf, 'InvertHardCopy', 'off'); %This retains the background color on save
	
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
	if c ~= 3;
		C = 4*h^2;
	else;
		C = 2*h^2; %gauss 2*std dev^2
	end
	xg = -5*h:h/2:5*h; 
	[Xg,Yg] = meshgrid(xg,xg);
	%wignersemicirc = (1/s)*sqrt(abs(s^2-Xg.^2-Yg.^2));
	gauss = exp(-(Xg.^2+Yg.^2)/C);
	fuzzyimage = conv2(locmask,gauss,'same');
	% %the convolution spreads things too much -- apply a threshold
	if c == 3;
		fuzzyimage = 1.9*fuzzyimage/max(max(fuzzyimage));
	% elseif c ==1;
	% 	fuzzyimage = 1.9*fuzzyimage/max(max(fuzzyimage));
	% elseif c == 2;
	else;
		fuzzyimage = 2*fuzzyimage/max(max(fuzzyimage));
	end
	mask1 = (fuzzyimage > 1);
	fuzzyimage(mask1) = 1;
	% mask = (fuzzyimage > 0.25);
	% fuzzyimage(mask) = 1;
	
	%get the stress and find the clims
	lx = squeeze(l(k,:,:,1));
	ly = squeeze(l(k,:,:,2));
	Str = squeeze(Strace(k,:,:));
	% surf(lx,ly,Str,'FaceColor','interp','EdgeColor','none','FaceLighting','phong')
	% clims = [cmin,cmax];
	
	%plot the star first, then the stress trace, then apply clims (only way I colud this to work -- gray magic)
	surf(X,Y,3*ones(size(X)),'FaceAlpha','interp','AlphaDataMapping','scaled','AlphaData',fuzzyimage,'FaceColor',[1,1,0],'EdgeColor','none')
	hold on
	surf(lx,ly,Str,'FaceColor','interp','EdgeColor','none','FaceLighting','phong')
	axis equal
	axis off
	colormap(Xmasmap)
	if c == 1;
		caxis([min(min(Str)),1.05*max(max(Str))]);
	elseif c == 2;
		caxis([min(min(Str)),max(max(Str))]);
	else;
		caxis([min(min(Str)),0.92*max(max(Str))])
	end
	view(2)	
	
	%now make the tree 
	
	for li = 1:(length(mpts(k,1:2:end))-2);
		if li <= 81;
			pt1 = [mpts(k,2*li-1),mpts(k,2*li)];
			pt2 = [mpts(k,2*li+3),mpts(k,2*li+4)];
			pt3 = [min(min(lx)),mpts(k,2*li+2)];
			pt4 = [min(min(lx)),mpts(k,2*li)];
			vx = [pt1(1),pt2(1),pt3(1),pt4(1)];
			vy = [pt1(2),pt2(2),pt3(2),pt4(2)];
			%patch(vx,vy,3*ones(size(vx)),3*ones(size(vx)),'FaceColor',bgcolor,'EdgeColor',bgcolor,'FaceVertexAlphaData',[0.5,1,1,1].','EdgeAlpha','flat')
			patch(vx,vy,3*ones(size(vx)),3*ones(size(vx)),'FaceColor',bgcolor,'EdgeColor',bgcolor)
		elseif li <= 81+36 || (li > 81+36+11+11+11 && li <= 81+36+11+11+11+36);
			pt1 = [mpts(k,2*li-1),mpts(k,2*li)];
			pt2 = [mpts(k,2*li+3),mpts(k,2*li+4)];
			pt3 = [mpts(k,2*li+3),min(min(ly))];
			pt4 = [mpts(k,2*li-1),min(min(ly))];
			vx = [pt1(1),pt2(1),pt3(1),pt4(1)];
			vy = [pt1(2),pt2(2),pt3(2),pt4(2)];
			% patch(vx,vy,3*ones(size(vx)),3*ones(size(vx)),'FaceColor',bgcolor,'EdgeColor',bgcolor,'FaceVertexAlphaData',[0.5,1,1,1].','EdgeAlpha','flat')
			patch(vx,vy,3*ones(size(vx)),3*ones(size(vx)),'FaceColor',bgcolor,'EdgeColor',bgcolor)
		elseif li > 81+36+11+11+11+36;
			pt1 = [mpts(k,2*li-1),mpts(k,2*li)];
			pt2 = [mpts(k,2*li+3),mpts(k,2*li+4)];
			pt3 = [mpts(k,2*li+3),max(max(lx))];
			pt4 = [mpts(k,2*li-1),max(max(lx))];
			vx = [pt1(1),pt2(1),pt3(1),pt4(1)];
			vy = [pt1(2),pt2(2),pt3(2),pt4(2)];
			% patch(vx,vy,3*ones(size(vx)),3*ones(size(vx)),'FaceColor',bgcolor,'EdgeColor',bgcolor,'FaceVertexAlphaData',[0.5,1,1,1].','EdgeAlpha','flat')
			patch(vx,vy,3*ones(size(vx)),3*ones(size(vx)),'FaceColor',bgcolor,'EdgeColor',bgcolor)
		end
		if any(any(ly > max(max(mpts(k,2:2:end)))));
			[ind,jnd] = find(ly >= max(max(mpts(k,2:2:end))));
			maxy = max(max(ly(ind,jnd)));
			miny = min(min(ly(ind,jnd)));
			maxx = max(max(lx(ind,jnd)));
			minx = min(min(lx(ind,jnd)));
			patch([minx,maxx/3,maxx/3,minx],[miny,miny,maxy,maxy],3*ones(4,1),3*ones(4,1),'FaceColor',bgcolor,'EdgeColor',bgcolor) 
			patch([3*maxx/4,maxx,maxx,3*maxx/4],[miny,miny,maxy,maxy],3*ones(4,1),3*ones(4,1),'FaceColor',bgcolor,'EdgeColor',bgcolor) 
		end
	end
	
end

