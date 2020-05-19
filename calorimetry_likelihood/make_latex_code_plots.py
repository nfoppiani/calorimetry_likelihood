import os
from glob import glob

filename = '/home/nic/Desktop/plots.tex'
base_folder = '/home/nic/Dropbox/MicroBooNE/_fig/calorimetry/'
out_folder = '/home/nic/Desktop/_fig/'

def add_figure(outfile, *fnames):
    # outfile.write('\\begin{center}\n')
    outfile.write('\\begin{figure}[H]\n')
    outfile.write('\\centering\n')
    n_plots = len(fnames)
    width = (1.-(n_plots-1)*0.05)/n_plots
    for fname in fnames:
        outfile.write('    \\subfloat[]{\\includegraphics[width='+'{:.3g}'.format(width)+'\\textwidth]{'+fname+'}}\\hfill\n')

    # if n_plots == 1:
    #     outfile.write('    \\centering\n')
    #     outfile.write('    \\includegraphics[width=1.00\\textwidth]{'+fnames[0]+'}\n')
    #     outfile.write('    \\caption{}\n')
    #     outfile.write('    \\label{fig:'+fnames[0]+'}\n')
    # else:
        
    #     for fname in fnames:
    #         outfile.write('    \\begin{subfigure}[b]{'+'{:.3g}'.format(width)+'\\textwidth}\n')
    #         outfile.write('    \\centering\n')
    #         outfile.write('    \\includegraphics[width=1.00\\textwidth]{'+fname+'}\n')
    #         outfile.write('    \\caption{}\n')
    #         outfile.write('    \\label{fig:'+fname+'}\n')
    #         outfile.write('    \\end{subfigure}\n')
    outfile.write('    \label{fig:'+fname+'}\n')
    outfile.write('    \caption{}\n')
    outfile.write('\\end{figure}\n')
    # outfile.write('\\end{center}\n')
    outfile.write('\\FloatBarrier\\n')

    for fname in fnames:
        os.system('cp --parents {} {}'.format(fname, out_folder))

if __name__ == "__main__":

    os.chdir(base_folder)
    out_f = open(filename, 'w')

    out_f.write('''\\section{ACPT}RE-calibration is performed using samples of ACPT in data and overlay.
In data Beam OFF, they are selected in the beam window, in the overlay they are simulated with Corsika in beam window, overlaid with events recorded with the unbiased trigger.
The selection is the following:
\\begin{itemize}
    \\item Track length > 20 cm
    \\item Track direction y < 0 (downward going)
\\end{itemize}
This has an efficiency of 97% in both data and simulation.
Additionally, for each cluster, the first and last hits are discarded.
The distributions of dE/dx in each bin of pitch and azimuthal angle $\\phi$ is compared between the data and the simulation.
The pitch and the azimuthal angle $\phi$ are computed consistently for each plane.
Except for the small Space Charge Correction, the pitch is a bijective function of the polar angle $\\theta$.
\\[ pitch = \\frac{0.3 \\text{cm}}{|\\cos(\\theta)|} \\]
With a binned maximum likelihood fit, the best multiplicative correction factor is obtained.
In this sense, the re-calibration is not a calibration, but a correction of the simulation to match the data.
''')

    out_f.write('\\subsection{Data/Monte Carlo before and after the recalibration}\n')
    out_f.write('''Comparison between data and simulation before and after the scale factor has been computed.
The sample has not been divided in train and test for these plots (i.e. train set = whole set), but it has been verified that there are no temporal differences within the samples and that by splitting the sample, you only increase the statistical fluctuations.
''')
    aux_data_mc_cali = [glob(base_folder+'acpt/data_mc_cali/plane{}/*.pdf'.format(i_pl)) for i_pl in [0, 1, 2]]
    data_mc_cali_lists = []
    for aux_file_list in aux_data_mc_cali:
        aux = [x.replace(base_folder, '') for x in aux_file_list]
        aux.sort()
        data_mc_cali_lists.append(aux)
    for fnames in zip(*data_mc_cali_lists):
        add_figure(out_f, *fnames)
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{Recalibration fits}\n')
    out_f.write('''Likelihood scans as a function of the scale factor in each bin of pitch and $\phi$.
The best value is obtained using scipy.optimize.minimize, but the likelihood scan is shown for clarity.
''')

    aux_calibration = [glob(base_folder+'acpt/calibration/plane{}/*.pdf'.format(i_pl)) for i_pl in [0, 1, 2]]
    calibration_lists = []
    for aux_file_list in aux_calibration:
        aux = [x.replace(base_folder, '') for x in aux_file_list]
        aux.sort()
        calibration_lists.append(aux)

    for fnames in zip(*calibration_lists):
        add_figure(out_f, *fnames)
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{Recalibration tables}\n')
    out_f.write('''The result of the recalibration is summarised in these tables that show the multiplicative factor in each bin of pitch and $\phi$.
There is one more bin at very large pitch which is not included in the plots.
''')
    for i_pl in [0, 1, 2]:
        add_figure(out_f, 'acpt/calibration/calibration_table_plane_{}.pdf'.format(i_pl))
    out_f.write('\\clearpage\\n')

    out_f.write('\section{Neutrino induced muons}\n')
    out_f.write('''Stopping muons produced by neutrino interactions are used to validate the re-calibration performed using ACPTs.
They are selected in data and simulation in the following way:
\\begin{itemize}
    \\item The longest track in the event
    \\item Track length > 30 cm
    \\item Track score > 0.5
    \\item Track start and end points are required to be contained in a fiducial volume, defined by 20 cm from all TPC borders. The start and end points are correct for SCE before applying the cut.
    \\item Track-vertex distance < 5 cm
    \\item Track-muon-momentum-consistency < 0.5. This variable ensures well reconstructed muons, where the range-based-momentum estimate agrees with the multiple-coulomb-scattering (MCS) estimate. The variable is defined as $\\displaystyle |p_{range} - p_{MCS}|/{p_range}$
\\end{itemize}
No calorimetry-based variable is used in the selection, in order not to bias the data-simulation comparisons.
Stopping muons are used two perform two main validations:
\\begin{itemize}
    \\item Residual range > 30 cm: MIP-like region. In this region data-simulation comparisons are performed in bins of pitch and $\phi$, before and after applying the correction. We expect to see similar discrepancies with respect to what observed with the ACPTs.
    \\item Residual range < 30 cm: Bragg-peak region. In this region data-simulation comparisons are performed in bins of residual range and pitch, before and after applying the correction. We want to ensure that the corrections derived on ACPTs, which are MIP particles, can be applied as well on larger energy depositions.
\\end{itemize}

The validation is performed by comparing the dQ/dx distribution from the data with the simulation before and after the recalibration.
The plots are filled once per hit, and they are normalised twice.
First the beam OFF is normalised to the beam ON data using the number of triggers, and it is then subtracted from the beam ON distribution.
Secondly, the simulation is weighted with the latest GENIE spline-times-tune weights, and then area normalised to the data beam ON - beam OFF.

It has been noticed that the corrections deried with the ACPTs do not work very well even with the MIP-like region of the stopping muons.
This means that the data-simulation discrepancies in the ACPTs are different than in the stopping muons.
In order to understand this difference, plots of the underlying distributions of pitch and $\\phi$ in the bins used for the recalibration are drawn.
In fact, the correction in bins of pitch and $\\phi$ is independent of the underlying distributions only in the approximation in which the distribution in every bin is flat.
As the bins are coarse, because of lack of statistics, the distributions cannot be approximated as flat in a bin, and thus, different distributions in the ACPTs versus MIP-muons lead to different data-simulation discrepancies in the dQ/dx distributions.
    ''')
    out_f.write('\subsection{Stopping muon selection}\n')
    aux_pot_norm = glob(base_folder+'muons/base_plots/plane{}/*pot.pdf'.format(i_pl))
    pot_norm = [x.replace(base_folder, '') for x in aux_pot_norm]
    pot_norm.sort()
    aux_area_norm = glob(base_folder+'muons/base_plots/plane{}/*area_norm.pdf'.format(i_pl))
    area_norm = [x.replace(base_folder, '') for x in aux_area_norm]
    area_norm.sort()
    for i, j in zip(pot_norm, area_norm):
        add_figure(out_f, i, j)
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{Data/Monte Carlo before and after the recalibration at residual range > 30 cm}\n')
    for i_pl in [0, 1, 2]:
        out_f.write('\subsubsection{Plane ' + '{}'.format(i_pl) + '}\n')
        aux_high_rr_before_after = glob(base_folder+'muons/high_rr_before_after/plane{}/*_noleg_*.pdf'.format(i_pl))
        high_rr_before_after = [x.replace(base_folder, '') for x in aux_high_rr_before_after]
        high_rr_before_after.sort()
        for i in range(len(high_rr_before_after)//2):
            add_figure(out_f, high_rr_before_after[2*i], high_rr_before_after[2*i + 1])
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{Summary of the effect of the recalibration}\n')
    for i_pl in [0, 1, 2]:
        out_f.write('\subsubsection{Plane ' + '{}'.format(i_pl) + '}\n')
        add_figure(out_f, *['muons/high_rr_before_after/plane_{}_chi2_{}.pdf'.format(i_pl, when) for when in ['avant', 'depois']])
        add_figure(out_f, *['muons/high_rr_before_after/plane_{}_{}.pdf'.format(i_pl, typ) for typ in ['delta_chi2', 'chi2_ratio']])
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{Data/Monte Carlo before and after the recalibration at residual range < 30 cm}\n')
    for i_pl in [0, 1, 2]:
        out_f.write('\subsubsection{Plane ' + '{}'.format(i_pl) + '}\n')
        aux_low_rr_before_after = glob(base_folder+'muons/low_rr_before_after/plane{}/*_noleg_*.pdf'.format(i_pl))
        low_rr_before_after = [x.replace(base_folder, '') for x in aux_low_rr_before_after]
        low_rr_before_after.sort()
        for i in range(len(low_rr_before_after)//2):
            add_figure(out_f, low_rr_before_after[2*i], low_rr_before_after[2*i + 1])
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{Summary of the effect of the recalibration}\n')
    for i_pl in [0, 1, 2]:
        out_f.write('\subsubsection{Plane ' + '{}'.format(i_pl) + '}\n')
        add_figure(out_f, *['muons/low_rr_before_after/plane_{}_chi2_{}.pdf'.format(i_pl, when) for when in ['avant', 'depois']])
        add_figure(out_f, *['muons/low_rr_before_after/plane_{}_{}.pdf'.format(i_pl, typ) for typ in ['delta_chi2', 'chi2_ratio']])
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{ACPT vs large residual range muons in the 2d pitch-$\phi$ distributions}\n')
    for i_pl in [0, 1, 2]:
        out_f.write('\subsubsection{Plane ' + '{}'.format(i_pl) + '}\n')
        aux_muons = glob(base_folder+'muons/pitch_phi/plane{}/*sim.pdf'.format(i_pl))
        muons = [x.replace(base_folder, '') for x in aux_muons]
        muons.sort()
        aux_acpt = glob(base_folder+'acpt/pitch_phi/plane{}/*.pdf'.format(i_pl))
        acpt = [x.replace(base_folder, '') for x in aux_acpt]
        acpt.sort()
        for i in range(len(muons)):
            add_figure(out_f, muons[i], acpt[i])
    out_f.write('\\clearpage\\n')

    out_f.write('\section{Neutrino induced protons}\n')

    out_f.write('\subsection{Stopping proton selection}\n')
    out_f.write('''Stopping protons produced by neutrino interactions are used to validate the re-calibration performed using ACPTs, specifically in the regions with large energy depositions.
They are selected in data and simulation in the following way:
\\begin{itemize}
    \\item Track score > 0.5
    \\item Track start and end points are required to be contained in a fiducial volume, defined by 20 cm from all TPC borders. The start and end points are correct for SCE before applying the cut.
    \\item Track-vertex distance < 5 cm
    \\item Track PID < -0.1
\\end{itemize}
Some distributions are shown before and after the PID cut.
This cut has a very large purity of protons, ensuring good coverage of the angular phase space.
The validation is performed by comparing the dQ/dx distribution from the data with the simulation before and after the recalibration, in bins of residual range and pitch.
The plots are normalised analogously to what explained for the stopping muons.
''')

    out_f.write('\subsubsection{Before PID cut}\n')
    aux_pot_norm = glob(base_folder+'proton/base_plots/plane{}/*before_pid_cut_pot.pdf'.format(i_pl))
    pot_norm = [x.replace(base_folder, '') for x in aux_pot_norm]
    pot_norm.sort()
    aux_area_norm = glob(base_folder+'proton/base_plots/plane{}/*before_pid_cut_area_norm.pdf'.format(i_pl))
    area_norm = [x.replace(base_folder, '') for x in aux_area_norm]
    area_norm.sort()
    for i, j in zip(pot_norm, area_norm):
        add_figure(out_f, i, j)
    out_f.write('\subsubsection{After PID cut}\n')
    aux_pot_norm = glob(base_folder+'proton/base_plots/plane{}/*after_pid_cut_pot.pdf'.format(i_pl))
    pot_norm = [x.replace(base_folder, '') for x in aux_pot_norm]
    pot_norm.sort()
    aux_area_norm = glob(base_folder+'proton/base_plots/plane{}/*after_pid_cut_area_norm.pdf'.format(i_pl))
    area_norm = [x.replace(base_folder, '') for x in aux_area_norm]
    area_norm.sort()
    for i, j in zip(pot_norm, area_norm):
        add_figure(out_f, i, j)
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{Data/Monte Carlo before and after the recalibration}\n')
    for i_pl in [0, 1, 2]:
        out_f.write('\subsubsection{Plane ' + '{}'.format(i_pl) + '}\n')
        aux_low_rr_before_after = glob(base_folder+'protons/low_rr_before_after/plane{}/*noleg*.pdf'.format(i_pl))
        low_rr_before_after = [x.replace(base_folder, '') for x in aux_low_rr_before_after]
        low_rr_before_after.sort()
        for i in range(len(low_rr_before_after)//2):
            add_figure(out_f, low_rr_before_after[2*i], low_rr_before_after[2*i + 1])
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{Summary of the effect of the recalibration}\n')
    for i_pl in [0, 1, 2]:
        out_f.write('\subsubsection{Plane ' + '{}'.format(i_pl) + '}\n')
        add_figure(out_f, *['protons/low_rr_before_after/plane_{}_chi2_{}.pdf'.format(i_pl, when) for when in ['avant', 'depois']])
        add_figure(out_f, *['protons/low_rr_before_after/plane_{}_{}.pdf'.format(i_pl, typ) for typ in ['delta_chi2', 'chi2_ratio']])

    out_f.write('\subsection{Data/Monte Carlo in the LLR PID variable}\n')
    aux_llr_pid_before_after = glob(base_folder+'protons/llr_pid/*_area_norm_noleg.pdf')
    llr_pid_before_after = [x.replace(base_folder, '') for x in aux_llr_pid_before_after]
    llr_pid_before_after.sort()
    for i in range(len(llr_pid_before_after)//3):
        add_figure(out_f, llr_pid_before_after[3*i], llr_pid_before_after[3*i + 1], llr_pid_before_after[3*i + 2])
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{Data/Monte Carlo with the systematic samples}\n')
    syst_samples = ['bnb_nu_mod', 'bnb_nu_recomb_mod', 'bnb_nu_wire_dedx_mod']
    for i_pl in [0, 1, 2]:
        out_f.write('\subsubsection{Plane ' + '{}'.format(i_pl) + '}\n')
        aux_low_rr = {sample:glob(base_folder+'protons_mod/low_rr/plane{}/{}*noleg.pdf'.format(i_pl, sample)) for sample in syst_samples}
        aux_low_rr = {sample:[x.replace(base_folder, '') for x in aux_low_rr[sample]] for sample in syst_samples}
        for file_list in aux_low_rr.values():
            file_list.sort()

        for i in range(len(file_list)):
            aux_plots = [aux_low_rr[sample][i] for sample in syst_samples]
            add_figure(out_f, *aux_plots)
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{Summary of the effect of the systematic samples}\n')
    reduced_syst_samples = ['bnb_nu_recomb_mod', 'bnb_nu_wire_dedx_mod']
    for i_pl in [0, 1, 2]:
        out_f.write('\subsubsection{Plane ' + '{}'.format(i_pl) + '}\n')
        add_figure(out_f, *['protons_mod/low_rr/{}_plane_{}_chi2.pdf'.format(sample, i_pl) for sample in syst_samples])
        add_figure(out_f, *['protons_mod/low_rr/{}_plane_{}_delta_chi2.pdf'.format(sample, i_pl) for sample in reduced_syst_samples])
        add_figure(out_f, *['protons_mod/low_rr/{}_plane_{}_chi2_ratio.pdf'.format(sample, i_pl) for sample in reduced_syst_samples])
    out_f.write('\\clearpage\\n')

    out_f.write('\section{Track PID}\n')
    out_f.write('''The track particle identification is performed using the likelihood ratio variable, which is the ratio of the likelihood under the proton and muon hypotheses.
In order to compute the likelihood, the probability density function (PDF) of the observed $dE/dx$ is built from the simulation, in bins of the residual range and pitch.
The PDFs for protons and muons are shown in the next section.
Performance plots are shown in the following one.

The tracks used to build the PDFs are selected in the following way:
\\begin{itemize}
    \\item Completeness > 0.9
    \\item Purity > 0.9
    \\item Track start and end points are required to be contained in a fiducial volume, defined by 20 cm from all TPC borders. The start and end points are correct for SCE before applying the cut.
    \\item No reconstructed PFP shr_daughter
    \\item The end-process of the particle at generator level must not be through inelastic scattering.
\\end{itemize}

As these conditions are partially based on truth level requirements, in order to evaluate the performance, a test set is obtained using only reconsutructed level requirements:
\\begin{itemize}
    \\item Track start and end points are required to be contained in a fiducial volume, defined by 20 cm from all TPC borders. The start and end points are correct for SCE before applying the cut.
    \\item No reconstructed PFP shr_daughter
\\end{itemize}
''')
    out_f.write('\subsection{Probability density functions}\n')
    aux_data_proton_muon_pdfs = [glob(base_folder+'proton_muon_likelihoods/pdfs/plane{}/*.pdf'.format(i_pl)) for i_pl in [0, 1, 2]]
    data_proton_muon_pdfs_lists = []
    for aux_file_list in aux_data_proton_muon_pdfs:
        aux = [x.replace(base_folder, '') for x in aux_file_list]
        aux.sort()
        data_proton_muon_pdfs_lists.append(aux)
    for fnames in zip(*data_proton_muon_pdfs_lists):
        add_figure(out_f, *fnames)
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{Performance plots}\n')
    out_f.write('\subsubsection{Plot of the variables}\n')
    llr_planes = ['proton_muon_likelihoods/performance_plots/llr_sum_{}_n.pdf'.format(i) for i in range(3)]
    add_figure(out_f, *llr_planes)
    llr_combined = ['proton_muon_likelihoods/performance_plots/llr_{}_n.pdf'.format(aux) for aux in ['01', '012']]
    add_figure(out_f, *llr_combined)
    out_f.write('\\clearpage\\n')

    out_f.write('\subsubsection{ROC curves}\n')
    add_figure(out_f, 'proton_muon_likelihoods/performance_plots/roc_curves.pdf')
    auc1d = ['proton_muon_likelihoods/performance_plots/auc1d_trk_{}.pdf'.format(aux) for aux in ['theta', 'phi']]
    add_figure(out_f, *auc1d)
    auc2d_planes = ['proton_muon_likelihoods/performance_plots/auc2dllr_sum_{}.pdf'.format(i) for i in range(3)]
    add_figure(out_f, *auc2d_planes)
    auc2d_combined = ['proton_muon_likelihoods/performance_plots/auc2dllr_{}.pdf'.format(aux) for aux in ['01', '012']]
    add_figure(out_f, *auc2d_combined)
    out_f.write('\\clearpage\\n')

    out_f.write('\section{Additional studies with the simulation}\n')
    out_f.write('\subsection{dE/dx vs residual range}\n')
    for i_pl in [0, 1, 2]:
        out_f.write('\subsubsection{Plane ' + '{}'.format(i_pl) + '}\n')
        aux_pdg_13 = glob(base_folder+'proton_muon_likelihoods/dedx_vs_rr/plane{}/pdg_13*.pdf'.format(i_pl))
        pdg_13 = [x.replace(base_folder, '') for x in aux_pdg_13]
        pdg_13.sort()
        aux_pdg_2212 = glob(base_folder+'proton_muon_likelihoods/dedx_vs_rr/plane{}/pdg_2212*.pdf'.format(i_pl))
        pdg_2212 = [x.replace(base_folder, '') for x in aux_pdg_2212]
        pdg_2212.sort()
        for i, j in zip(pdg_13, pdg_2212):
            add_figure(out_f, i, j)
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{Hit $\cos \\theta$ vs $\phi$}\n')
    for i_pl in [0, 1, 2]:
        out_f.write('\subsubsection{Plane ' + '{}'.format(i_pl) + '}\n')
        aux_pdg_13 = glob(base_folder+'proton_muon_likelihoods/hit_cos_theta_phi/plane{}/pdg_13*.pdf'.format(i_pl))
        pdg_13 = [x.replace(base_folder, '') for x in aux_pdg_13]
        pdg_13.sort()
        add_figure(out_f, *pdg_13)
        aux_pdg_2212 = glob(base_folder+'proton_muon_likelihoods/hit_cos_theta_phi/plane{}/pdg_2212*.pdf'.format(i_pl))
        pdg_2212 = [x.replace(base_folder, '') for x in aux_pdg_2212]
        pdg_2212.sort()
        add_figure(out_f, *pdg_2212)
    out_f.write('\\clearpage\\n')

    out_f.write('\subsubsection{Track $\cos \\theta$ vs $\phi$}\n')
    aux_pdg_13 = glob(base_folder+'proton_muon_likelihoods/cos_theta_phi/pdg_13*.pdf'.format(i_pl))
    pdg_13 = [x.replace(base_folder, '') for x in aux_pdg_13]
    pdg_13.sort()
    add_figure(out_f, *pdg_13)
    aux_pdg_2212 = glob(base_folder+'proton_muon_likelihoods/cos_theta_phi/pdg_2212*.pdf'.format(i_pl))
    pdg_2212 = [x.replace(base_folder, '') for x in aux_pdg_2212]
    pdg_2212.sort()
    add_figure(out_f, *pdg_2212)
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{Hit $\cos \\theta$ vs pitch}\n')
    for i_pl in [0, 1, 2]:
        out_f.write('\subsubsection{Plane ' + '{}'.format(i_pl) + '}\n')
        aux_pdg_13 = glob(base_folder+'proton_muon_likelihoods/hit_cos_theta_pitch/plane{}/pdg_13*.pdf'.format(i_pl))
        pdg_13 = [x.replace(base_folder, '') for x in aux_pdg_13]
        pdg_13.sort()
        add_figure(out_f, *pdg_13)
        aux_pdg_2212 = glob(base_folder+'proton_muon_likelihoods/hit_cos_theta_pitch/plane{}/pdg_2212*.pdf'.format(i_pl))
        pdg_2212 = [x.replace(base_folder, '') for x in aux_pdg_2212]
        pdg_2212.sort()
        add_figure(out_f, *pdg_2212)
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{Fraction of bad hits ($dE/dx \leq 1$ MeV/cm)}\n')
    for i_pl in [0, 1, 2]:
        out_f.write('\subsubsection{Plane ' + '{}'.format(i_pl) + '}\n')
        aux_pdg_13 = glob(base_folder+'proton_muon_likelihoods/bad_hit_fraction_cos_theta_phi/plane{}/pdg_13*.pdf'.format(i_pl))
        pdg_13 = [x.replace(base_folder, '') for x in aux_pdg_13]
        pdg_13.sort()
        add_figure(out_f, *pdg_13)
        aux_pdg_2212 = glob(base_folder+'proton_muon_likelihoods/bad_hit_fraction_cos_theta_phi/plane{}/pdg_2212*.pdf'.format(i_pl))
        pdg_2212 = [x.replace(base_folder, '') for x in aux_pdg_2212]
        pdg_2212.sort()
        add_figure(out_f, *pdg_2212)
    out_f.write('\\clearpage\\n')

    out_f.write('\section{Additional studies with the data Beam ON}\n')
    out_f.write('\subsection{dE/dx vs residual range in bins of pitch}\n')
    for i_pl in [0, 1, 2]:
        out_f.write('\subsubsection{Plane ' + '{}'.format(i_pl) + '}\n')
        aux_pdg_muon = glob(base_folder+'beam_on_dedx_rr/pitch/plane{}/pdg_muon*.pdf'.format(i_pl))
        pdg_muon = [x.replace(base_folder, '') for x in aux_pdg_muon]
        pdg_muon.sort()
        aux_pdg_proton = glob(base_folder+'beam_on_dedx_rr/pitch/plane{}/pdg_proton*.pdf'.format(i_pl))
        pdg_proton = [x.replace(base_folder, '') for x in aux_pdg_proton]
        pdg_proton.sort()
        for i, j in zip(pdg_muon, pdg_proton):
            add_figure(out_f, i, j)
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{dE/dx vs residual range in bins of track pitch Y}\n')
    for i_pl in [0, 1, 2]:
        out_f.write('\subsubsection{Plane ' + '{}'.format(i_pl) + '}\n')
        aux_pdg_muon = glob(base_folder+'beam_on_dedx_rr/track_pitch_y/plane{}/pdg_muon*.pdf'.format(i_pl))
        pdg_muon = [x.replace(base_folder, '') for x in aux_pdg_muon]
        pdg_muon.sort()
        aux_pdg_proton = glob(base_folder+'beam_on_dedx_rr/track_pitch_y/plane{}/pdg_proton*.pdf'.format(i_pl))
        pdg_proton = [x.replace(base_folder, '') for x in aux_pdg_proton]
        pdg_proton.sort()
        for i, j in zip(pdg_muon, pdg_proton):
            add_figure(out_f, i, j)
    out_f.write('\\clearpage\\n')

    out_f.write('\section{Shower PID}\n')
    out_f.write('''The attempt to the shower particle identification is performed using an analogous method to the one developed for tracks.
In the following, while refering to showers, I will refer to the track-fit objects.
These objects have anab::calorimetry objects associated, and thus, all the technology developed for tracks can be applied to the showers as well.
In this case, the likelihood ratio variable is the ratio of the likelihood under the electron and photon hypotheses.
The probability density function (PDF) of the observed $dE/dx$ is built from the simulation, in bins of the distance from the shower start-point and pitch.
The PDFs for electrons and photons are shown in the next section.
Performance plots are shown in the following one.

The showers used to build the PDFs are selected in the following way:
\\begin{itemize}
    \\item Purity > 0.9
    \\item The start point is required to be contained in a fiducial volume, defined by 20 cm from all TPC borders. The start and end points are correct for SCE before applying the cut.
    \\item The 3D distance between the shower stat point and the first point where energy is deposited, corrected for SCE, is required to be smaller than 2 cm.
    \\item The track-fit object is required to have at least one hit on any of the three planes. This condition is non trivial as it might happen that the track-fit rejects all the hits of a given PFParticle.
\\end{itemize}

As these conditions are partially based on truth level requirements, in order to evaluate the performance, a test set is obtained using only the reconsutructed level requirements:
\\begin{itemize}
    \\item The start point is required to be contained in a fiducial volume, defined by 20 cm from all TPC borders. The start and end points are correct for SCE before applying the cut.
    \\item The track-fit object is required to have at least one hit on any of the three planes. This condition is non trivial as it might happen that the track-fit rejects all the hits of a given PFParticle.
\\end{itemize}
''')
    out_f.write('\subsection{Preliminary studies with the showers}\n')

    start_point_res = ['shower_calo/base_plots/electron_photon_startpoint_res{}.pdf'.format(aux) for aux in ['', '_zoom']]
    add_figure(out_f, *start_point_res)

    dedx_2d = ['shower_calo/base_plots/{}_2d_dedx_dist.pdf'.format(aux) for aux in ['electron', 'photon']]
    add_figure(out_f, *dedx_2d)

    aux = glob(base_folder+'shower_calo/base_plots/photon_2d_dedx_dist_sh_energy*.pdf')
    aux = [x.replace(base_folder, '') for x in aux]
    aux.sort()
    add_figure(out_f, *aux)

    add_figure(out_f, 'shower_calo/base_plots/photon_1d_dedx_sh_energy.pdf', 'shower_calo/base_plots/electron_photon_shower_energy.pdf')

    out_f.write('\subsection{Probability density functions}\n')
    aux_data_shower_calo_pdfs = [glob(base_folder+'shower_calo/pdfs/plane{}/*.pdf'.format(i_pl)) for i_pl in [0, 1, 2]]
    data_shower_calo_pdfs_lists = []
    for aux_file_list in aux_data_shower_calo_pdfs:
        aux = [x.replace(base_folder, '') for x in aux_file_list]
        aux.sort()
        data_shower_calo_pdfs_lists.append(aux)
    for fnames in zip(*data_shower_calo_pdfs_lists):
        add_figure(out_f, *fnames)
    out_f.write('\\clearpage\\n')

    out_f.write('\subsection{Performance plots}\n')
    out_f.write('\subsubsection{Plot of the variables}\n')
    dedx_vars = ['shower_calo/performance_plots/dedx_median_4_{}_n.pdf'.format(plane) for plane in ['u', 'v', 'y']]
    add_figure(out_f, *dedx_vars)
    llr_vars = ['shower_calo/performance_plots/llr_sum_{}_n.pdf'.format(i_pl) for i_pl in [0, 1, 2]]
    add_figure(out_f, *llr_vars)
    llr_combined = ['shower_calo/performance_plots/llr_{}_n.pdf'.format(aux) for aux in ['01', '012']]
    add_figure(out_f, *llr_combined)
    out_f.write('\\clearpage\\n')

    out_f.write('\subsubsection{Plot of the variables in bins of true shower energy}\n')
    for var in ['dedx_median_y', 'llr_sum_2', 'llr_01', 'llr_012']:
        aux = glob(base_folder+'shower_calo/performance_plots/{}_n_sh_energy*.pdf'.format(var))
        aux = [x.replace(base_folder, '') for x in aux]
        aux.sort()
        for i in range(len(aux)//2):
            add_figure(out_f, aux[2*i], aux[2*i + 1])
    out_f.write('\\clearpage\\n')

    out_f.write('\subsubsection{ROC curves}\n')
    add_figure(out_f, 'shower_calo/performance_plots/roc_curves.pdf', 'shower_calo/performance_plots/auc1d_backtracked_e.pdf')
    out_f.write('\\clearpage\\n')
    
    out_f.close()
