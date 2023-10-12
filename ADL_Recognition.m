clc;
clear;
close all;

%load subject 101-109 , select your needs and save as tables
for i=1:9
    eval(['load(''/Users/dengliu2000/Documents/MATLAB/PAMAP2-Protocol/subject10',num2str(i),'.dat'');'])
    eval(['subject10',num2str(i),'_HandChestAnkle_AccGyro(:,1:3)=subject10',num2str(i),'(:,5:7);'])
    eval(['subject10',num2str(i),'_HandChestAnkle_AccGyro(:,4:6)=subject10',num2str(i),'(:,11:13);'])
    eval(['subject10',num2str(i),'_HandChestAnkle_AccGyro(:,7:9)=subject10',num2str(i),'(:,22:24);'])
    eval(['subject10',num2str(i),'_HandChestAnkle_AccGyro(:,10:12)=subject10',num2str(i),'(:,28:30);'])
    eval(['subject10',num2str(i),'_HandChestAnkle_AccGyro(:,13:15)=subject10',num2str(i),'(:,39:41);'])
    eval(['subject10',num2str(i),'_HandChestAnkle_AccGyro(:,16:18)=subject10',num2str(i),'(:,45:47);'])
    eval(['subject10',num2str(i),'_HandChestAnkle_AccGyro(:,19)=subject10',num2str(i),'(:,2);'])
end

%window size, sliding point, feature table and ground table
WS=100;
SP=50;
for i=1:9
eval(['NumofWindow=fix(((length(subject10',num2str(i),')-WS)/SP)+1);'])
for j=1:NumofWindow
    interval=1+(j-1)*50:100+(j-1)*50;
    for k=1:18
    eval(['FT_subject10',num2str(i),'(j,1+(k-1)*6)=mean(subject10',num2str(i),'_HandChestAnkle_AccGyro(interval,1));'])
    eval(['FT_subject10',num2str(i),'(j,2+(k-1)*6)=std(subject10',num2str(i),'_HandChestAnkle_AccGyro(interval,1));'])
    eval(['FT_subject10',num2str(i),'(j,3+(k-1)*6)=max(subject10',num2str(i),'_HandChestAnkle_AccGyro(interval,1));'])
    eval(['FT_subject10',num2str(i),'(j,4+(k-1)*6)=min(subject10',num2str(i),'_HandChestAnkle_AccGyro(interval,1));'])
    eval(['FT_subject10',num2str(i),'(j,5+(k-1)*6)=range(subject10',num2str(i),'_HandChestAnkle_AccGyro(interval,1));'])
    eval(['FT_subject10',num2str(i),'(j,6+(k-1)*6)=var(subject10',num2str(i),'_HandChestAnkle_AccGyro(interval,1));'])
    eval(['FT_subject10',num2str(i),'_HandChestAnkle_AccGyro(:,19)=subject10',num2str(i),'(:,2);'])
    eval(['GT_subject10',num2str(i),'(j,1)=mode(subject10',num2str(i),'_HandChestAnkle_AccGyro(interval,19));'])
    eval(['FT_GT_subject10',num2str(i),'=[FT_subject10',num2str(i),',GT_subject10',num2str(i),'];'])
    end
end
end

% create a table to save
FT_GT=[];
data_traintest_matrix=[];
for i=1:9
    eval(['FT_GT=[FT_GT;FT_GT_subject10',num2str(i),'];'])
    eval(['L',num2str(i),'=length(FT_GT_subject10',num2str(i),');'])
    eval(['data_traintest_matrix_L',num2str(i),'(1:L',num2str(i),',1)=i;'])
    eval(['data_traintest_matrix=[data_traintest_matrix;data_traintest_matrix_L',num2str(i),'];'])

end

%leave one out cross validation, confusionmatrix(KNN, DT, NB)
for i=1:9
    test=(data_traintest_matrix==i);
    train=~test;
    FT_GT_test=FT_GT(test,:);
    FT_GT_train=FT_GT(train,:);
    modelKNN=fitcknn(FT_GT_train(:,1:108),FT_GT_train(:,109),"NumNeighbors",3);
    modelNB=fitcnb(FT_GT_train(:, 1:108), FT_GT_train(:,109));
    modelDT=fitctree(FT_GT_train(:, 1:108), FT_GT_train(:,109));
    predictmodelKNN=predict(modelKNN,FT_GT_test(:,1:108));
    predictmodelNB=predict(modelNB, FT_GT_test(:, 1:108));
    predictmodelDT=predict(modelDT, FT_GT_test(:, 1:108));
    eval(['confusionmatrix_KNN_',num2str(i),'=confusionmat(FT_GT_test(:,109),predictmodelKNN,''Order'',[0 1 2 3 4 5 6 7 12 13 16 17 24]);'])
    eval(['confusionmatrix_NB_',num2str(i),'=confusionmat(FT_GT_test(:,109),predictmodelNB,''Order'',[0 1 2 3 4 5 6 7 12 13 16 17 24]);'])
    eval(['confusionmatrix_DT_',num2str(i),'=confusionmat(FT_GT_test(:,109),predictmodelDT,''Order'',[0 1 2 3 4 5 6 7 12 13 16 17 24]);'])
    
    % calculate Acc, Sen, Pre (window by window's result)
    eval(['TotalMatrix_W1(i,1)=sum(diag(confusionmatrix_KNN_',num2str(i),'))/sum(sum(confusionmatrix_KNN_',num2str(i),'));'])
    eval(['TotalMatrix_W2(i,1)=sum(diag(confusionmatrix_NB_',num2str(i),'))/sum(sum(confusionmatrix_NB_',num2str(i),'));'])
    eval(['TotalMatrix_W3(i,1)=sum(diag(confusionmatrix_DT_',num2str(i),'))/sum(sum(confusionmatrix_DT_',num2str(i),'));'])
    eval(['TotalMatrix_W1(i,2)=confusionmatrix_KNN_',num2str(i),'(i,i)/sum(confusionmatrix_KNN_',num2str(i),'(i,:));'])
    eval(['TotalMatrix_W2(i,2)=confusionmatrix_NB_',num2str(i),'(i,i)/sum(confusionmatrix_NB_',num2str(i),'(i,:));'])
    eval(['TotalMatrix_W3(i,2)=confusionmatrix_DT_',num2str(i),'(i,i)/sum(confusionmatrix_DT_',num2str(i),'(i,:));'])
    eval(['TotalMatrix_W1(i,3)=confusionmatrix_KNN_',num2str(i),'(i,i)/sum(confusionmatrix_KNN_',num2str(i),'(:,i));'])
    eval(['TotalMatrix_W2(i,3)=confusionmatrix_NB_',num2str(i),'(i,i)/sum(confusionmatrix_NB_',num2str(i),'(:,i));'])
    eval(['TotalMatrix_W3(i,3)=confusionmatrix_DT_',num2str(i),'(i,i)/sum(confusionmatrix_DT_',num2str(i),'(:,i));'])
    
    % calculate Acc, Sen, Pre (sample by sample's result), not much
    % difference between them
    
    predictpoint_KNN=[];
    predictpoint_NB=[];
    predictpoint_DT=[];
    
    for j=1:length(predictmodelKNN)
        predictpoint_KNN(1+(j-1)*SP:100+(j-1)*SP,1)=predictmodelKNN(j,1);
    end
    eval(['predictpoint_KNN(length(predictpoint_KNN)+1:length(subject10',num2str(i),'),1)=predictmodelKNN(length(predictmodelKNN),1);'])
    for j=1:length(predictmodelNB)
        predictpoint_NB(1+(j-1)*SP:100+(j-1)*SP,1)=predictmodelNB(j,1);
    end
    eval(['predictpoint_NB(length(predictpoint_NB)+1:length(subject10',num2str(i),'),1)=predictmodelNB(length(predictmodelNB),1);'])
    for j=1:length(predictmodelDT)
        predictpoint_DT(1+(j-1)*SP:100+(j-1)*SP,1)=predictmodelDT(j,1);
    end
    eval(['predictpoint_DT(length(predictpoint_DT)+1:length(subject10',num2str(i),'),1)=predictmodelDT(length(predictmodelDT),1);'])
    
    eval(['confusionmatrix_KNN_',num2str(i),'=confusionmat(subject10',num2str(i),'_HandChestAnkle_AccGyro(:,19),predictpoint_KNN,''Order'',[0 1 2 3 4 5 6 7 12 13 16 17 24]);'])
    eval(['confusionmatrix_NB_',num2str(i),'=confusionmat(subject10',num2str(i),'_HandChestAnkle_AccGyro(:,19),predictpoint_NB,''Order'',[0 1 2 3 4 5 6 7 12 13 16 17 24]);'])
    eval(['confusionmatrix_DT_',num2str(i),'=confusionmat(subject10',num2str(i),'_HandChestAnkle_AccGyro(:,19),predictpoint_DT,''Order'',[0 1 2 3 4 5 6 7 12 13 16 17 24]);'])
    eval(['TotalMatrix_S1(i,1)=sum(diag(confusionmatrix_KNN_',num2str(i),'))/sum(sum(confusionmatrix_KNN_',num2str(i),'));'])
    eval(['TotalMatrix_S2(i,1)=sum(diag(confusionmatrix_NB_',num2str(i),'))/sum(sum(confusionmatrix_NB_',num2str(i),'));'])
    eval(['TotalMatrix_S3(i,1)=sum(diag(confusionmatrix_DT_',num2str(i),'))/sum(sum(confusionmatrix_DT_',num2str(i),'));'])
    eval(['TotalMatrix_S1(i,2)=confusionmatrix_KNN_',num2str(i),'(i,i)/sum(confusionmatrix_KNN_',num2str(i),'(i,:));'])
    eval(['TotalMatrix_S2(i,2)=confusionmatrix_NB_',num2str(i),'(i,i)/sum(confusionmatrix_NB_',num2str(i),'(i,:));'])
    eval(['TotalMatrix_S3(i,2)=confusionmatrix_DT_',num2str(i),'(i,i)/sum(confusionmatrix_DT_',num2str(i),'(i,:));'])
    eval(['TotalMatrix_S1(i,3)=confusionmatrix_KNN_',num2str(i),'(i,i)/sum(confusionmatrix_KNN_',num2str(i),'(:,i));'])
    eval(['TotalMatrix_S2(i,3)=confusionmatrix_NB_',num2str(i),'(i,i)/sum(confusionmatrix_NB_',num2str(i),'(:,i));'])
    eval(['TotalMatrix_S3(i,3)=confusionmatrix_DT_',num2str(i),'(i,i)/sum(confusionmatrix_DT_',num2str(i),'(:,i));'])
end