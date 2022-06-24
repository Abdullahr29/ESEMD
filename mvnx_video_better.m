%% load mvnx
[mvnxFileName,pathName] = uigetfile('/input_data/matforvideo');
tree = importdata([pathName mvnxFileName]); 

segments=tree.subject.segments.segment;
joints = tree.subject.joints.joint;
data = tree.subject.frames.frame(4:end);

skl = [data.position];
skl = reshape(skl,3,[],numel(data));
skl = permute(skl,[3 2 1]);
skl = skl-skl(:,1,:);

%need to add line to open error.mat for joint with highest error vector

%% Plot Skeleton
eventIX = 3978;
% plot first frame
frame=eventIX;
figure('Color',[1 1 1]); 
h_s = scatter3(skl(frame,:,1),skl(frame,:,2),skl(frame,:,3)); 
hold on;
plot3(skl(frame,11,1),skl(frame,11,2),skl(frame,11,3),'ro','MarkerFaceColor','r')
xlim([min(min(skl(:,:,1))) max(max(skl(:,:,1)))]); 
ylim([min(min(skl(:,:,2))) max(max(skl(:,:,2)))]); 
zlim([min(min(skl(:,:,3))) max(max(skl(:,:,3)))]);
xlabel('X (a.u.)');
ylabel('Y (a.u.)');
zlabel('Z (a.u.)');

linIX = {[1:7],[5 8:11],[5 12:15],[1 16:19],[1 20:23]}; 
clear h_l
for n=1:numel(linIX)
    h_l(n) = plot3(skl(frame,linIX{n},1),skl(frame,linIX{n},2),skl(frame,linIX{n},3));
end

% view(122,6)
% F = getframe;
% fIX=2;
% 
% v = VideoWriter([pathName mvnxFileName '.mp4'],'MPEG-4');
% v.FrameRate=60;
% open(v);
% 
% % movie
% 
% while 1
%     try
%         h_s.XData = skl(frame,:,1);
%     catch
%         break
%     end
%     h_s.XData = skl(frame,:,1);
%     h_s.YData = skl(frame,:,2);
%     h_s.ZData = skl(frame,:,3);    
%     for n=1:numel(linIX)
%         h_l(n).XData=skl(frame,linIX{n},1);
%         h_l(n).YData=skl(frame,linIX{n},2);
%         h_l(n).ZData=skl(frame,linIX{n},3);
%     end 
%     title(mvnxFileName)
%     drawnow;
%     videoFrame = getframe;
%     writeVideo(v,videoFrame);
% %     pause(.01)
% %     F(fIX) = getframe;
% %     fIX=fIX+1;
%     frame = frame + 1;
% end
% close(v)
% %% Plot Skelaton HC2
% eventIX = 175;
% % plot first frame
% frame=eventIX;
% figure('Color',[1 1 1]); h_s = scatter3(skl(frame,:,1),skl(frame,:,2),skl(frame,:,3)); hold on; 
% linIX = {[1:7],[5 8:11],[5 12:15],[1 16:19],[1 20:23]}; clear h_l
% for n=1:numel(linIX); h_l(n) = plot3(skl(frame,linIX{n},1),skl(frame,linIX{n},2),skl(frame,linIX{n},3)); end
% 
% set(gca,'xTickLabel',[],'yTickLabel',[],'zTickLabel',[])
% xlim([min(min(skl(:,:,1))) max(max(skl(:,:,1)))]); 
% ylim([min(min(skl(:,:,2))) max(max(skl(:,:,2)))]); 
% zlim([min(min(skl(:,:,3))) max(max(skl(:,:,3)))]);  view(-58,6)
% F = getframe; fIX=2;
% 
% % movie
% for frame=eventIX+[70:125]
%     h_s.XData = skl(frame,:,1);   h_s.YData = skl(frame,:,2);    h_s.ZData = skl(frame,:,3);    
%     for n=1:numel(linIX); h_l(n).XData=skl(frame,linIX{n},1); h_l(n).YData=skl(frame,linIX{n},2); h_l(n).ZData=skl(frame,linIX{n},3); end 
%     title(mvnxFileName)
%     drawnow; pause(.01)
%     F(fIX) = getframe; fIX=fIX+1;
% end


%% movie file
% v = VideoWriter([pathName mvnxFileName '.mp4'],'MPEG-4');
% v.FrameRate=60;  open(v); writeVideo(v,F);



