import pickle
import settings
def compute_theta(iteration):
  ksi=settings.ksi
  g=settings.g
  H=settings.H
  K=settings.K
  kappa=settings.kappa
  nigijk=settings.nigijk
  nigij=settings.nigij
  alpha=settings.alpha
  nih=settings.nih
  N=settings.N
  theta=settings.theta
  result_path=settings.result_path
  ############
  ############
  
  for i in range(0,N):
    for h in range(0,H):
        summation=sum(ksi[i][h])
        for k in range(0,K):
          denominator=((summation*alpha) + nigij[i][h])
          if denominator==0:
            theta[i][h][k]=0
          else:
            theta[i][h][k]=((ksi[i][h][k]*alpha)+ nigijk[i][h][k]) / denominator
  
  theta_path=result_path+"theta_"+str(iteration)
  with open(theta_path, 'wb') as L2:
      pickle.dump(theta,L2)
      
  settings.theta=theta 