using LightGraphs
using SimpleWeightedGraphs
using LinearAlgebra
using Random

function cc(dataset,p,epsilon,alpha_0)

    matA=dataset';
    p=p';#100x1
    alpha_01=zeros(length(alpha_0),1);
    for i in 1:length(alpha_0)
        alpha_01[i,1]=alpha_0[i];
    end
    alpha_0=alpha_01;#(k,1)
    SmatA=size(matA);
    m=SmatA[1];#100
    n=SmatA[2];#k
    pp=zeros(m,n);
    for i in 1:n
        pp[:,i]=p;#100x1--100Xk
    end
    diffmat=matA-pp;
    diffmat2=diffmat'*diffmat;
    diffmn=size(diffmat2)
    diffm=diffmn[1];
    eudis = Vector(undef, diffm);
    for i in 1:diffm
        eudis[i]=sqrt(diffmat2[i,i]);
    end
    min_index=argmin(eudis);
    p_pp=matA[:,min_index];
    p_p=zeros(m,1);
    for i in 1:m
        p_p[i,1]=p_pp[i];
    end
    alpha=zeros(1,n);
    alpha[1,min_index]=1;
    if size(alpha_0)[1]==n && n>1
        rcal=size(alpha_0);
        ral=rcal[1];
        cal=rcal[2];
        if ral>cal
            alpha_0=alpha_0';
        end
        for i in 1:n
            alpha[1,i]=alpha_0[1,i]
        end
        p_p=matA*alpha';
    end
    distance=eudis[min_index];
    if n<=1
        distance=eudis[min_index];
        inorout=0;
        p_prime1=p_p;
        alpha_coe1=alpha;
        dist1=distance;
        return inorout,p_prime1,alpha_coe1,dist1;
    end
    inorout=1;
    dist_vp=sum(diffmat2,dims=1);
    beta_list=zeros(n,1);

    thre = sqrt((p-p_p)'*(p-p_p))[1,1] - 5
    while sqrt((p-p_p)'*(p-p_p))[1,1]>thre
        found=0;
        att=(p-p_p)'*(p-p_p);
        distance = sqrt(att);
        mnp_p=size(p_p);
        p_p2=zeros(mnp_p[1],n);
        for i in 1:n
            p_p2[:,i]=p_p;
        end
        ppv=p_p2-matA;
        norm2_ppv=sum(ppv.^2,dims=1);
        norm2_ppvab=size(norm2_ppv);
        norm_ppv=zeros(norm2_ppvab[1],norm2_ppvab[2]);
        for i in 1:norm2_ppvab[2]
            norm_ppv[1,i]=sqrt(norm2_ppv[1,i]);
        end
        gd=matA'*(p-p_p);#kx100*100x1=kx1
        p_norm=p'*p;
        p_p_norm=p_p'*p_p;
        dist_diff=ones(size(gd)[1],1);
        for i in 1:size(gd)[1]
            dist_diff[i,1]=(p_norm-p_p_norm)[1]-gd[i,1]*2;
        end
        index_pivot=[];
        for i in 1:size(dist_diff)[1]
            if dist_diff[i,1]<=0
                push!(index_pivot,i);
            end
        end
        gd2=zeros(size(gd)[1],1);
        for i in 1:size(gd)[1]
            gd2[i,:]=p_p'*(p_p-p)+gd[i,:];
        end
        pre_beta_list=gd2./norm2_ppv';
        for i in 1:size(pre_beta_list)[1]
            if pre_beta_list[i,1]==Inf
                pre_beta_list[i,1]=0;
            end
            if pre_beta_list[i,1]==-Inf
                pre_beta_list[i,1]=0;
            end
        end
        if length(index_pivot)==0
            found=0;
        else
            geq_index=[];
            leq_index=[];
            for i in 1:size(pre_beta_list)[1]
                if pre_beta_list[i,1]>0
                    push!(geq_index,i);
                end
            end
            for i in 1:size(pre_beta_list)[1]
                if pre_beta_list[i,1]<0
                    push!(leq_index,i);
                end
            end
            pre_beta_listnew=[];
            for i in geq_index
                push!(pre_beta_listnew,pre_beta_list[i,1])
            end
            push!(pre_beta_listnew,1);
            a_newmin=sort(pre_beta_listnew);
            mina=[];
            for i in 1:length(a_newmin)
                push!(mina,i)
            end
            if length(mina)!=0
                beta_list[geq_index].=a_newmin[mina[1]];
            end
            if length(leq_index)!=0
                alpha2=(alpha[leq_index])./(alpha[leq_index].-ones(length(leq_index)));#(1xk)
                alpha3=pre_beta_list[leq_index];#(kx1)
                for i in 1:length(leq_index)
                    push!(alpha3,alpha2[i]);
                end
                alpha3=sort(alpha3);
                maxalpha3=alpha3[length(alpha3)];
                beta_list[leq_index].=maxalpha3;
            end
            beta_list2=zeros(size(beta_list)[1],size(beta_list)[2]);
            for i in 1:size(beta_list)[1]
                beta_list2[i,1]=abs(beta_list[i,1]);
            end
            this_len=beta_list2.*norm_ppv';
            v_index=argmax(this_len);
            beta=beta_list[v_index[1]];
            alpha=(1-beta)*alpha;
            alpha[v_index[1]]=alpha[v_index[1]]+beta;
            p_p=(1-beta)*p_p+beta*matA[:,v_index[1]];
            found=1;
            if beta<=1e-8
                bata=0;
                found=0;
            end
        end
        if found==0
            inorout=0;
            break;
        end
    end
    p_prime=p_p;
    alpha_coe=alpha;
    dist=distance;
    return inorout,p_prime,alpha_coe,dist;
end