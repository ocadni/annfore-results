import numpy as np

def make_diff(margs_diff,margs_sib, t, norm_I):
    norm_I_temp=margs_diff[:,t,1].sum()
    if np.any(np.isnan(margs_diff)):
        print("NAN marginals: ", np.isnan(margs_diff).sum()/np.prod(margs_diff.shape))
    elif norm_I_temp == 0:
        #print("ZERO SUM")
        raise ValueError("Zero sum")
    m_nn = margs_diff[:,t,1]/norm_I_temp
    diff = np.abs(m_nn - margs_sib).sum()
#print(f"{str_name_file}, {diff/I}")
    merr = diff/(norm_I)

    return merr


def make_convergence(ress, seeds, num_conf, lambda_, steps, Ns, key_analy="regressive", reps=None):
    """
    @author: Indaco Biazzo
    """
    t = 0 #list(range(t_limit))
    conver = []
    num_plot = 0
    for i_seed, seed in enumerate(seeds):

            #plt.subplot(2, 3, cl+1)
        for instance_num in range(num_conf[i_seed]):
            for cl in range(len(lambda_)):
                I = 0
                res_case = ress[seed][lambda_[cl]][instance_num]

                I = res_case["sib"]["marginals"][:,-1, 1].sum()
                #norm_I = ress[seed][Ns[cl]][instance_num]["sib"]["marginals"][:,:, 1].sum()
                m_sib = res_case["sib"]["marginals"][:,t,1]
                norm_I = m_sib.sum()#I * t_limit
                #print(norm_I)
                #print(steps)
                for i_step, step in enumerate(steps):

                    str_name_file = str(step)
                    if reps is None:
                        margs = res_case[key_analy][step]["marginals"]
                        merr = make_diff(margs, m_sib, t, norm_I=norm_I)

                        conver.append({
                            "step":step,
                            "err":merr,
                            "N":Ns[cl],
                            "I":I
                        })
                    else:
                        for r in reps:
                            margs = res_case[key_analy][step][r]["marginals"]
                            merr = make_diff(margs, m_sib, t, norm_I=norm_I)

                            conver.append({
                            "step":step,
                            "err":merr,
                            "N":Ns[cl],
                            "I":I,
                            "rep": r,
                            })
    
    return conver