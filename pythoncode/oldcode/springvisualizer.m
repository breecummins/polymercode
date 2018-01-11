%springvisualizer.m

clear
close all

set(0,'DefaultAxesFontSize',24)

%%%%%%%%%%%%%%%%%%%%%%  Two head spring %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
titlestring='Wi = 6.4';
fnamestring='064';

stokesfile = '~/rsyncfolder/data/VEflow/TwoHeadSpring/springtestC_stokes.mat';
vefile = ['~/rsyncfolder/data/VEflow/TwoHeadSpring/springtestC_visco', fnamestring, '.mat'] ;


load(stokesfile)
% plot(t,fpts(:,1),'r','LineWidth',2)
% hold on
% plot(t,fpts(:,3),'m','LineWidth',2)
stokest=t;
distbwsprings = sqrt((fpts(:,1)-fpts(:,3)).^2 + (fpts(:,2)-fpts(:,4)).^2);
% axis([t(1),t(end),fpts(1,1)-0.05,fpts(1,3)+0.05])
% xlabel('Time')
% ylabel('x-coordinate')

load(vefile)
% plot(t,fpts(:,1),'b','LineWidth',2)
% plot(t,fpts(:,3),'c','LineWidth',2)
% legend('Stokes','Stokes','VE','VE')
% hold off

figure
plot(stokest,distbwsprings,'r','LineWidth',2)
hold on
plot(t,sqrt((fpts(:,1)-fpts(:,3)).^2 + (fpts(:,2)-fpts(:,4)).^2),'b','LineWidth',2)
xlabel('Time')
ylabel('Distance between beads')
legend('Stokes','VE')

pause(1.0)

%max and min stress
set(0,'DefaultAxesFontSize',12)

figure
if gcf == 1;
	load(vefile)
end
S11=[]; S12=[]; S21=[]; S22=[];
for k =1:length(t);
	S11(1,k) = max(max(S(k,:,:,1,1)));
	S11(2,k) = min(min(S(k,:,:,1,1)));
	S12(1,k) = max(max(S(k,:,:,1,2)));
	S12(2,k) = min(min(S(k,:,:,1,2)));
	S21(1,k) = max(max(S(k,:,:,2,1)));
	S21(2,k) = min(min(S(k,:,:,2,1)));
	S22(1,k) = max(max(S(k,:,:,2,2)));
	S22(2,k) = min(min(S(k,:,:,2,2)));
end

	
subplot(2,2,1)
plot(t,S11(1,:),'k.')
hold on
plot(t,S11(2,:),'k.')
title('S11')
MM = 1.05*max(S11(1,:));
mm = 0.95*min(S11(2,:));
axis([0,t(end),mm,MM])
grid on

%S12
subplot(2,2,2)
plot(t,S12(1,:),'k.')
hold on
plot(t,S12(2,:),'k.')
title('S12')
MM = 1.05*max(S12(1,:));
mm = 1.05*min(S12(2,:));
axis([0,t(end),mm,MM])
grid on

%S21
subplot(2,2,3)
plot(t,S21(1,:),'k.')
hold on
plot(t,S21(2,:),'k.')
xlabel('Time')
title('S21')
MM = 1.05*max(S21(1,:));
mm = 1.05*min(S21(2,:));
axis([0,t(end),mm,MM])
grid on

%S22
subplot(2,2,4)
plot(t,S22(1,:),'k.')
hold on
plot(t,S22(2,:),'k.')
xlabel('Time')
title('S22')
MM = 1.05*max(S22(1,:));
mm = 0.95*min(S22(2,:));
axis([0,t(end),mm,MM])
grid on

% figureSize = get(gcf,'Position');
% uicontrol('Style','text','String','Min and max stress values','Position',[(figureSize(3)-100)/2 figureSize(4)-25 100 25],'BackgroundColor',get(gcf,'Color'));
% 
%%%%%%%%%%%%%%%%%%
set(0,'DefaultAxesFontSize',24)

figure
if gcf == 1;
	load(vefile)
end
set(gcf, 'PaperSize', [8 6]);
set(gcf, 'PaperPosition', [0,0,8,6]);
c=0;
MM = 1.05*max(max(max(Strace)));
mm = 0.95*min(min(min(Strace)));
for k = 1:length(t)  
	lx = squeeze(l(k,:,:,1));
	ly = squeeze(l(k,:,:,2));
	Str = squeeze(Strace(k,:,:));
	%scatter(lx(:),ly(:),[],Str(:),'.')  
	surf(lx,ly,Str,'EdgeColor','k','facecolor','interp')
	axis([-0.2,1.2,-0.2,1.2])
	colormap(cool)
	colorbar
	caxis([mm,MM])
	caxis([1.9,2.9])
	hold on 
	plot3(fpts(k,[1,3]),fpts(k,[2,4]),[3,3],'k.','MarkerSize',24)
	hold off
	title([titlestring, ', Time = ',num2str(t(k))])
	c=c+1;
	axis off
	% pause(0.001)
	% if mod(k,10) == 1;	
		saveas(gcf,['~/scratch/frame_twohead_Wi',fnamestring,'_',sprintf('%03d',c),'.png'],'png')
	% else;
	% 	pause(0.001)
	% end
	
	if t(k) > 0.5 && max(max(Str)) < 2.02;
		disp(t(k))
	end
end




%%%%%%%%%%%%%%%%%%%  simple spring %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load ~/scratch/teststokesflow.mat
% plot(t,sqrt(fpts(:,1).^2 + fpts(:,2).^2),'r','LineWidth',2)
% hold on
% load ~/scratch/testveflow.mat
% plot(t,sqrt(fpts(:,1).^2 + fpts(:,2).^2),'b','LineWidth',2)
% hold off
% xlabel('Time')
% ylabel('Distance from the origin')
% 
% pause(5.0)

% figure
% if gcf == 1;
% 	load ~/scratch/testveflow_noforces.mat
% end
% set(gcf, 'PaperSize', [8 6]);
% set(gcf, 'PaperPosition', [0,0,8,6]);
% N = 150;
% c=0;
% for k = 1:length(t)  
% 	if k < N || mod(k,10) == 0;
% 		lx = l(k,:,:,1);
% 		ly = l(k,:,:,2);
% 		scatter(lx(:),ly(:),[],Ptrace(k,:))  
% 		axis([-0.5,1.5,-0.5,1.5])
% 		colorbar
% 		caxis([min(1.9,min(Ptrace(k,:))),max(2.1,max(Ptrace(k,:)))])
% 		hold on 
% 		plot(fpts(k,1),fpts(k,2),'k.')
% 		x = -0.5:0.1:1.5;
% 		plot(x,zeros(size(x)),'k')
% 		plot(zeros(size(x)),x,'k')
% 		hold off
% 		title(['Time = ',num2str(t(k))])
% 		c=c+1;
% 		saveas(gcf,['~/scratch/frame_noforces',sprintf('%03d',c),'.eps'],'eps')
% 	end
% end
% 

