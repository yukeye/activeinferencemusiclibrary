%% Set up concept model
%__________________________________________________________________________
clear
close all
rng('shuffle')
%rng('default')

%% Simulation options

%set to 1 to:

%simulate multiple trials (as opposed to only 1)

trial_sequence = 1;

%remove knowledge (must enable A-matrix learning) - only choose 1 option
%-------------------------------------------------------------------------
likelihood_A_learning = 1; %enable A-matrix learning

%prevent reporting/feedback
prevent_reporting = 0;

remove_all_knowledge = 0; %all song concepts


%remove specific song knowledge
remove_song1_knowledge = 0;
remove_song2_knowledge = 0;
remove_song3_knowledge = 0;
remove_song4_knowledge = 0;
remove_song5_knowledge = 0;
remove_song6_knowledge = 0;



%perform bayesian model reduction (must enable D-matrix learning)
%-------------------------------------------------------------------------
prior_D_learning = 1; %enable -matrix learning: 1 = enable, 0 = disable

BMR = 1; %enable -Bayesian model reduction: 1 = enable, 0 = disable

%choose which concepts to include: 1 = include, 0 = remove
D{1} = [1 1 1 1 0 1]'; % concepts: {'song1','song2','song3','song4','song5','song6'}


%% generative process and prior beliefs about initial states (in terms of counts: D and d
%--------------------------------------------------------------------------
if BMR == 0
D{1} = [1 1 1 1 1 1]';           % concept:     {'song1','song2','song3','song4','song5','song6'} total: 6
end

D{2} = [1 0 0 0 0 0 0]'; % report:    {'start','Report song1'...'report song6'} total: 7

d{1} = [1 1 1 1 1 1]'; % total: 6
d{2} = D{2}; 

%% mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------
Nf    = numel(D); % number of factors
for f = 1:Nf
    Ns(f) = numel(D{f});
end
No    = [6 3]; % number of outcomes per modality (song1-6, feedback)
Ng    = numel(No);
for g = 1:Ng
    A{g} = zeros([No(g),Ns]);
end


%A  matrix
%--------------------------------------------------------------------------


for i = 1:Ns(2) % number of states in each hidden states
    A{1}(:,:,i) = [1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1]; %Song 1 to 6
end



A{2}(:,:,1) = [1 1 1 1 1 1; 0 0 0 0 0 0; 0 0 0 0 0 0]; 
A{2}(:,:,2) = [0 0 0 0 0 0; 1 0 0 0 0 0; 0 1 1 1 1 1]; 
A{2}(:,:,3) = [0 0 0 0 0 0; 0 1 0 0 0 0; 1 0 1 1 1 1]; 
A{2}(:,:,4) = [0 0 0 0 0 0; 0 0 1 0 0 0; 1 1 0 1 1 1]; 
A{2}(:,:,5) = [0 0 0 0 0 0; 0 0 0 1 0 0; 1 1 1 0 1 1]; 
A{2}(:,:,6) = [0 0 0 0 0 0; 0 0 0 0 1 0; 1 1 1 1 0 1]; 
A{2}(:,:,7) = [0 0 0 0 0 0; 0 0 0 0 0 1; 1 1 1 1 1 0]; 
 


for g = 1:Ng
    A{g} = double(A{g});
end
           
%% beliefs about state-outcome mappings: a

a = A;

pa = 0; % inverse temperature parameter; 0 = completely flatten state-outcome mapping distributions for concepts
           
if remove_all_knowledge == 1

%remove all concept (song) knowledge
    a{1}(1:6,:,:)=spm_softmax(pa*log(a{1}(1:6,:,:)+exp(-4)))+ .01*randn(size(a{1}(1:6,:,:))); % only content precision
    
    
end


%remove concept of songs 1 to 6

if remove_song1_knowledge == 1
    a{1}(1:6,1,:)=spm_softmax(pa*log(a{1}(1:6,1,:)+exp(-4)))+ .01*randn(size(a{1}(1:6,1,:))); % only content precision
    
   
end

if remove_song2_knowledge == 1
    a{1}(1:6,2,:)=spm_softmax(pa*log(a{1}(1:6,2,:)+exp(-4)))+ .01*randn(size(a{1}(1:6,2,:))); % only content precision
    
   
end

if remove_song3_knowledge == 1
    a{1}(1:6,3,:)=spm_softmax(pa*log(a{1}(1:6,3,:)+exp(-4)))+ .01*randn(size(a{1}(1:6,3,:))); % only content precision
    
   
end

if remove_song4_knowledge == 1
    a{1}(1:6,4,:)=spm_softmax(pa*log(a{1}(1:6,4,:)+exp(-4)))+ .01*randn(size(a{1}(1:6,4,:))); % only content precision
    
   
end

if remove_song5_knowledge == 1
    a{1}(1:6,5,:)=spm_softmax(pa*log(a{1}(1:6,5,:)+exp(-4)))+ .01*randn(size(a{1}(1:6,5,:))); % only content precision
    
   
end

if remove_song6_knowledge == 1
    a{1}(1:6,6,:)=spm_softmax(pa*log(a{1}(1:6,6,:)+exp(-4)))+ .01*randn(size(a{1}(1:6,6,:))); % only content precision
    
   
end



%% controlled transitions
%--------------------------------------------------------------------------
for f = 1:Nf
    B{f} = eye(Ns(f));
end
 
% controllable report state
%--------------------------------------------------------------------------
for k = 1:Ns(2)
    B{2}(:,:,k) = 0;
    B{2}(k,:,k) = 1;
end

B{2}(:,2:7,:) = 0;

for i = 2:7
    B{2}(i,i,:) = 1;
end


%% allowable policies (here, specified as the next action) U
%--------------------------------------------------------------------------
T = 2;
Np        = size(B{2},3);
U         = ones(1,Np-1,Nf);
U(:,:,2)  = 2:Np;


if prevent_reporting == 1

%prevent reporting
%--------------------------------------------------------------------------

%U = [];
U(:,:,1)  = [1];
U(:,:,2)  = [1]; % not controllable

end

%% prior preferences: C
%--------------------------------------------------------------------------
C{1}      = zeros(No(1),T);

C{2}      = zeros(No(2),T);
C{2}(2,:) =   4;                 % Correct
C{2}(3,:) =  -4;                 % Incorrect




%% MDP Structure - this will be used to generate arrays for multiple trials
%==========================================================================
mdp.T = T;                      % number of moves
mdp.V = U;                     % allowable shallow policies
mdp.A = A;                      % observation model
mdp.B = B;                      % transition probabilities
mdp.C = C;                      % preferred outcomes
mdp.D = D;                      % prior over initial states

if likelihood_A_learning == 1
mdp.a = a;                      % observation beliefs
end

if prior_D_learning == 1
mdp.d = d;                     % prior beliefs over initial states
end

label.factor{1}   = 'song-concept';   label.name{1}    = {'song1','song2','song3','song4','song5','song6'};
label.factor{2}   = 'self-reports';     label.name{2}    = {'start','report song1','report song2','report song3','report song4','report song5','report song6'};
label.modality{1} = 'song-identity';    label.outcome{1} = {'ID_of_song1','ID_of_song2','ID_of_song3','ID_of_song4','ID_of_song5','ID_of_song6' };
label.modality{2} = 'feedback';    label.outcome{3} = {'start','correct','incorrect'};
label.actions{1,1} = ' ';
label.actions{1,2} = ' ';
mdp.label = label;

mdp.alpha = 128;% inverse temperature for action selection
mdp.beta = 1;% prior policy precision

mdp         = spm_MDP_check(mdp);

%% illustrate a single trial
%==========================================================================
MDP   = spm_MDP_VB_X(mdp);

% show belief updates (and behaviour)
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 1'); clf
spm_MDP_VB_trial(MDP(1), 1:2,2);
subplot(3,2,3)


if trial_sequence == 1
%% illustrate a sequence of trials
%==========================================================================
clear MDP


N = 100;% number of learning trials
 
for i = 1:N
   MDP(i)   = mdp;      % create structure array
end

% Solve - an example sequence
%==========================================================================
MDP  = spm_MDP_VB_X(MDP);
 
% illustrate behavioural responses and neuronal correlates
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 4'); clf
spm_MDP_VB_game(MDP(1:20: N));
 
spm_figure('GetWin','Figure 6'); clf
spm_MDP_VB_LFP(MDP(1:20: N)); 

% plot A, and a (before and after learning)

if likelihood_A_learning == 1

LMA = [MDP(N).A{1}(1:6,:,1)];
LMa = [MDP(N).a{1}(1:6,:,1)];
LMa1 = [mdp.a{1}(1:6,:,1)]; 
  

spm_figure('GetWin','generative process'); clf
imagesc(LMA), colormap gray % generative process

spm_figure('GetWin','generative model after learning'); clf
imagesc(LMa), colormap gray % after learning

spm_figure('GetWin','generative model before learning'); clf
imagesc(LMa1), colormap gray % before learning
end


end

%% Bayesian model reduction
if BMR == 1
if prior_D_learning == 1

qA = MDP(N).d{1};
pA = d{1};
rA = [permn([1 6], 6)]';

spm_figure('GetWin','posterior distribution'); clf
imagesc(qA'), colormap gray

spm_figure('GetWin','prior distribution'); clf
imagesc(pA'), colormap gray

spm_figure('GetWin','true distribution'); clf
imagesc([MDP(N).D{1}]'), colormap gray

evidence = [];
      
for i = 1:size(rA,1)
[evidence] = [evidence spm_MDP_log_evidence(qA,pA,rA)];
end

[Min, idx] = min(evidence);
winner_evidence = Min
winner = [rA(:,idx)']

spm_figure('GetWin','winning model'); clf
imagesc(winner), colormap gray

end
end

