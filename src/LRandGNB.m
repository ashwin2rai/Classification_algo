
clear; clc;
hold on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code compares Gaussin Naiive Bayes and 
% Logistic Regression classifiers for a dataset
% with 4 features and 1 class label performing
% binary classification. The code uses vectorization
% and broadcasting where possible to speed up 
% the algorithms and uses gradient descent for 
% weight optimzation in the case of Logistic 
% Regression. 
%
% The code trains the classifiers with different
% proportions of the total data to show underfitting
% and overfitting and biased predictions. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%loading dataset: may have to change directories
load data_bn;
%data_bn=importdata('iris.data');

tra=0;
temp_var=find(data_bn(:,5)==1
data_bn(temp_var),1)= data_bn(find(temp_var),1) + tra;
ctr=0;

%Initial weights
w = [ 0.1 0.1 0.1 0.1 0.1]';

tic;

%proportion of data that will be used for training per run
training_data_per_v=.01:.005:1;
repmax=5;

for newvar = 1:length(training_data_per_v)
    
    for rep=1:repmax
        ctr=ctr+1;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Data Preprocessing
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        training_data_per=training_data_per_v(newvar);
        data_bn_random = data_bn(randperm(length(data_bn)),:);
        training_data =data_bn_random(1:ceil(training_data_per*...
            length(data_bn)),:);
        testing_set = data_bn(randperm(length(data_bn)),:);
        training_data_bn=sortrows(training_data,5);
        dummyvar=find(training_data_bn(:,5)==1);
        zero_data=training_data_bn(1:dummyvar(1)-1,:);
        one_data=training_data_bn(dummyvar,:);
        
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Guassian Naiive Bayes MLE Parameter estimation
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        mu_p=zeros(8,1);
        vars_p=zeros(8,1);
	
	%initializing for speed
        length_zd=size(zero_data);
        length_zd=length_zd(1,1);
        length_od=size(one_data);
        length_od=length_od(1,1);
        
        for i=1:4
            mu_p(i) = sum(zero_data(:,i))/(length_zd);
            vars_p(i) =   sum((zero_data(:,i) - ...
                mu_p(i)*ones(length_zd,1)).^2)/...
                (length_zd-1);
            
            mu_p(i+4) = sum(one_data(:,i))/(length_od);
            vars_p(i+4) =   sum((one_data(:,i) - ...
                mu_p(i+4)*ones(length_od,1)).^2)/...
                (length_od-1);
        end
        
        p_0 = (length_zd)/(length(training_data));
        p_1 = 1-p_0;
        
        prob_0 = (p_0)...
            .* (normpdf(testing_set(:,1),mu_p(1),sqrt(vars_p(1)))) ...
            .* (normpdf(testing_set(:,2),mu_p(2),sqrt(vars_p(2)))) ...
            .* (normpdf(testing_set(:,3),mu_p(3),sqrt(vars_p(3)))) ...
            .* (normpdf(testing_set(:,4),mu_p(4),sqrt(vars_p(4))));
        
        prob_1 = (p_1)...
            .* (normpdf(testing_set(:,1),mu_p(5),sqrt(vars_p(5)))) ...
            .* (normpdf(testing_set(:,2),mu_p(6),sqrt(vars_p(6)))) ...
            .* (normpdf(testing_set(:,3),mu_p(7),sqrt(vars_p(7)))) ...
            .* (normpdf(testing_set(:,4),mu_p(8),sqrt(vars_p(8))));
        
        predict_vec=(prob_1>prob_0);
	
	%accuracy metrics
        error_gnb(ctr)=sum( predict_vec ~= testing_set(:,5))...
            /length(testing_set);
        tr_length(ctr)=length(training_data);
        
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Logistic Regression - training weights
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        eta = 0.005;
        err = 1;
        x0 = ones(length(training_data),1);
        x1 = training_data(:,1);
        x2 = training_data(:,2);
        x3 = training_data(:,3);
        x4 = training_data(:,4);
        y  = training_data(:,5);
        while err > 1e-3
            
            add_var=0;
            for j=1:length(x1)
                dum_var=exp(w(1)+ w(2)*x1(j) + w(3)*x2(j)+ w(4)*x3(j)...
                    + w(5)*x4(j));
                add_var= add_var + x0(j)*((dum_var)/(1+dum_var));
            end
            wnew1= w(1) + eta*(y'*x0 - add_var);
            
            add_var=0;
            for j=1:length(x1)
                dum_var=exp(w(1)+ w(2)*x1(j) + w(3)*x2(j)+ w(4)*x3(j)...
                    + w(5)*x4(j));
                add_var= add_var + x1(j)*((dum_var)/(1+dum_var));
            end
            wnew2= w(2) + eta*(y'*x1 - add_var);
            
            
            add_var=0;
            for j=1:length(x1)
                dum_var=exp(w(1)+ w(2)*x1(j) + w(3)*x2(j)+ w(4)*x3(j)...
                    + w(5)*x4(j));
                add_var= add_var + x2(j)*((dum_var)/(1+dum_var));
            end
            wnew3= w(3) + eta*(y'*x2 - add_var);
            
            add_var=0;
            for j=1:length(x1)
                dum_var=exp(w(1)+ w(2)*x1(j) + w(3)*x2(j)+ w(4)*x3(j)...
                    + w(5)*x4(j));
                add_var= add_var + x3(j)*((dum_var)/(1+dum_var));
            end
            wnew4= w(4) + eta*(y'*x3 - add_var);
            
            add_var=0;
            for j=1:length(x1)
                dum_var=exp(w(1)+ w(2)*x1(j) + w(3)*x2(j)+ w(4)*x3(j)...
                    + w(5)*x4(j));
                add_var= add_var + x4(j)*((dum_var)/(1+dum_var));
            end
            wnew5= w(5) + eta*(y'*x4 - add_var);
            
            err=mean(w-[wnew1;wnew2;wnew3;wnew4;wnew5;]);
            w=[wnew1;wnew2;wnew3;wnew4;wnew5;];
            
        end
        
        predict_vec_l= w(1)*ones(length(testing_set),1) + ...
            w(2)*testing_set(:,1)+ w(3)*testing_set(:,2) + ...
            w(4)*testing_set(:,3) +  w(5)*testing_set(:,4);
        predict_vec = predict_vec_l>0;
	
	%finding accuracy metrics
        error_lr(ctr)=sum( predict_vec ~= testing_set(:,5))...
            /length(testing_set);
	%%
    end
end
toc;

tr_length=mean(reshape(tr_length,[repmax length(training_data_per_v) ]));
error_gnb=mean(reshape(error_gnb,[repmax length(training_data_per_v) ]));
error_lr=mean(reshape(error_lr,[repmax length(training_data_per_v) ]));
figure(1);
plot(tr_length./length(data_bn),error_gnb);
hold on;
plot(tr_length./length(data_bn),error_lr);
axis([0 1 -0.1 0.5])

figure(2);
hold on;
plot(tra,mean(error_gnb),'rs');
plot(tra,mean(error_lr),'bs');
