import pickle
import settings
def compute_BETA(iteration):
  kappa=settings.kappa
  nkj=settings.nkj
  K=settings.K
  V=settings.V
  BETA=settings.BETA
  nk=settings.nk
  result_path=settings.result_path
  phikw=settings.phikw
    
  for k in range(0,K):
    for w in range(0,V):
      try:
          BETA[k][w] = ((phikw[k][w]*kappa) + nkj[k][w]) / ( (sum(phikw[k])*kappa) + nk[k] )
      except:
          BETA[k][w]=0
  BETA_path=result_path+"BETA_"+str(iteration)

  with open(BETA_path, 'wb') as L3:
      pickle.dump(BETA,L3)
  if settings.image_dataset_flag:
      print("BETA")
      for bk in BETA:
          print(bk)    
  settings.BETA=BETA
