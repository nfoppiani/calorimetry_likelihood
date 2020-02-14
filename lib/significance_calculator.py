import numpy as np
from matplotlib import pyplot as plt
import uproot
import mpmath
import pdb
from likelihoods import LEff, LEff_v


class significanceCalculator(object):

    def __init__(self, expected_bin_contents, bin_edges=None, pot=1):
        if expected_bin_contents is not None:
            assert 'signal' in expected_bin_contents.keys()
            if type(list(expected_bin_contents.values())[0]) == np.ndarray:
                self.expected_bin_contents = expected_bin_contents
                self.expected_bin_sigmas = None
                self.n_bin = len(expected_bin_contents['signal'])
            elif isinstance(list(expected_bin_contents.values())[0], tuple) or isinstance(list(expected_bin_contents.values())[0], list):
                bin_means = [value[0] for value in expected_bin_contents.values()]
                bin_sigmas = [value[2] for value in expected_bin_contents.values()]
                self.expected_bin_contents = dict(zip(expected_bin_contents.keys(), bin_means))
                self.expected_bin_sigmas = dict(zip(expected_bin_contents.keys(), bin_sigmas))
                self.n_bin = len(expected_bin_contents['signal'][0])
            else:
                print('something went wrong')
                return

            self.bin_edges = bin_edges

            self.total_bg_prediction = np.zeros(self.n_bin)
            for name, prediction in self.expected_bin_contents.items():
                if name == 'signal':
                    continue
                self.total_bg_prediction += prediction

            if self.expected_bin_sigmas is not None:
                self.total_bg_sigma2 = np.zeros(self.n_bin)
                self.signal_sigma2 = self.expected_bin_sigmas['signal']**2
                if self.expected_bin_sigmas is not None:
                    for name, sigma in self.expected_bin_sigmas.items():
                        if name == 'signal':
                            continue
                        self.total_bg_sigma2 += sigma**2

        self.pot = pot

        self.ts_dict = {'pois_llr': self.neymanPearsonLikelihoodRatio,
                        'gaus_llr': self.gaussianLikelihoodRatio,
                        'delta_chi2': self.deltaChi2,
                        'chi2_mu0': self.Chi2_mu0,
                        'chi2_mu1': self.Chi2_mu1,
                        'eff_llr': self.logRatioEffectiveLikelihood,
                        }

        self.ts_labels = {'pois_llr': 'Poisson log likelihood ratio',
                          'gaus_llr': 'Gaussian log likelihood ratio',
                          'delta_chi2': r'$X^2(\mu=1) - X^2(\mu=0)$',
                          'chi2_mu0': r'$X^2(\mu=0)$',
                          'chi2_mu1': r'$X^2(\mu=1)$',
                          'eff_llr': 'Effective log likelihood ratio',
                          }

    def buildFromDataframes(self, dict_dataframe, bin_edges, nu_oscillator, deltam2, sin2theta2):
        self.dict_dataframe = dict_dataframe
        self.bin_edges = bin_edges
        self.n_bin = len(bin_edges) - 1
        self.expected_bin_contents = {'bg': np.zeros(self.n_bin),
                  'nue': np.zeros(self.n_bin),
                  'signal': np.zeros(self.n_bin)}
        self.expected_bin_sigmas = None
        for sample_name, sample_df in self.dict_dataframe.items():
            if sample_name == 'on':
                continue
            sample_df['weight'] = sample_df['weightSpline']
            if sample_name in ['nu', 'dirt']:
                sample_df['weight'] = sample_df['weightSpline'] * (~sample_df['nueccinc'])
                sample_df['signal_weight'] = sample_df['weight'] * nu_oscillator.oscillationWeightAtEnergy(sample_df['nu_e'], deltam2, sin2theta2)
            elif sample_name == 'nue':
                sample_df['weight'] = sample_df['weightSpline'] * (sample_df['nueccinc'])
                sample_df['signal_weight'] = sample_df['weight'] * nu_oscillator.oscillationWeightAtEnergy(sample_df['nu_e'], deltam2, sin2theta2)
            elif sample_name == 'off':
                sample_df['signal_weight'] = sample_df['weightSpline'] * 0
                sample_df['nu_pdg'] = 0

            aux_mean_bg, _ = np.histogram(sample_df['shr_energy_y_v']/1000 * (np.abs(sample_df['nu_pdg']) != 12),
                                          weights=sample_df['weight']*(np.abs(sample_df['nu_pdg']) != 12),
                                          bins=bin_edges)
            aux_mean_nue, _ = np.histogram(sample_df['shr_energy_y_v']/1000 * (np.abs(sample_df['nu_pdg']) == 12),
                                          weights=sample_df['weight']*(np.abs(sample_df['nu_pdg']) == 12),
                                          bins=bin_edges)
            aux_signal, _ = np.histogram(sample_df['shr_energy_y_v']/1000 * (np.abs(sample_df['nu_pdg']) == 12),
                                          weights=sample_df['signal_weight']*(np.abs(sample_df['nu_pdg']) == 12),
                                          bins=bin_edges)
            # import pdb; pdb.set_trace();
            self.expected_bin_contents['bg'] += aux_mean_bg
            self.expected_bin_contents['nue'] += aux_mean_nue
            self.expected_bin_contents['signal'] += aux_signal
        self.total_bg_prediction = self.expected_bin_contents['bg'] + self.expected_bin_contents['nue']

    def setDataFromTTrees(self, dict_tfile_names, dict_pot_scaling, weight_variable='weightSplineTimesTune'):
        self.dict_dataset = {}
        for sample_name, tfile_name in dict_tfile_names.items():
            aux_file = uproot.open(tfile_name)
            aux_array = aux_file['NeutrinoSelectionFilter'].arrays(namedecode="utf-8")
            if weight_variable in aux_array.keys():
                aux_array['weight'] = aux_array[weight_variable] * dict_pot_scaling[sample_name]
            else:
                aux_array['weight'] = (aux_array['evt'] == aux_array['evt']) * dict_pot_scaling[sample_name]
            self.dict_dataset[sample_name] = aux_array

    def setVariableOfInterest(self, bin_edges, reco_energy, true_energy, true_pdg):
        self.bin_edges = bin_edges
        self.reco_energy = reco_energy
        self.n_bin = len(bin_edges) - 1
        self.true_energy = true_energy
        self.true_pdg = true_pdg

        for array in self.dict_dataset.values():
            if true_energy not in array.keys():
                array[true_energy] = 0*(aux_array['evt'] == aux_array['evt'])
            if true_pdg not in array.keys():
                array[true_pdg] = 0*(aux_array['evt'] == aux_array['evt'])

    def setNuOscillator(self, nu_oscillator, deltam2, sin2theta2):
        self.nu_oscillator = nu_oscillator
        self.deltam2 = deltam2
        self.sin2theta2 = sin2theta2
        self.label = (r'Sensitivity to $\Delta m^2_{14}$ = ' +
                  '{:.3g}'.format(deltam2) +
                  r' $eV^2, \sin^2(2\theta_{e\mu})$ = ' +
                  '{:.3g}'.format(sin2theta2))

    def setPOT(self, pot):
        self.pot = pot

    def setSelectionLabel(self, label):
        self.selection_label = label

    def fillHistogramOsc(self):
        self.expected_bin_contents = {'bg': np.zeros(self.n_bin),
                  'nue': np.zeros(self.n_bin),
                  'signal': np.zeros(self.n_bin)}
        for sample_name, array in self.dict_dataset.items():
            array['signal_weight'] = array['weight'] * self.nu_oscillator.oscillationWeightAtEnergy(array['nu_e'], self.deltam2, self.sin2theta2)

            inf_mask = ~np.isinf(array['weight'])
            nu_e = (np.abs(array[self.true_pdg]) == 12) & inf_mask
            non_nu_e = (np.abs(array[self.true_pdg]) != 12) & inf_mask

            aux_mean_bg, _ = np.histogram(array[self.reco_energy][non_nu_e],
                                          weights=array['weight'][non_nu_e],
                                          bins=self.bin_edges)
            aux_mean_nue, _ = np.histogram(array[self.reco_energy][nu_e],
                                          weights=array['weight'][nu_e],
                                          bins=self.bin_edges)
            aux_signal, _ = np.histogram(array[self.reco_energy][nu_e],
                                          weights=array['signal_weight'][nu_e],
                                          bins=self.bin_edges)

            self.expected_bin_contents['bg'] += aux_mean_bg
            self.expected_bin_contents['nue'] += aux_mean_nue
            self.expected_bin_contents['signal'] += aux_signal
        self.total_bg_prediction = self.expected_bin_contents['bg'] + self.expected_bin_contents['nue']

    def fillHistogramNueBeam(self):
        self.expected_bin_contents = {'bg': np.zeros(self.n_bin),
                  'nue': np.zeros(self.n_bin),
                  'signal': np.zeros(self.n_bin)}
        for sample_name, array in self.dict_dataset.items():
            inf_mask = ~np.isinf(array['weight'])
            nu_e = (np.abs(array[self.true_pdg]) == 12) & inf_mask
            non_nu_e = (np.abs(array[self.true_pdg]) != 12) & inf_mask

            aux_mean_bg, _ = np.histogram(array[self.reco_energy][non_nu_e],
                                          weights=array['weight'][non_nu_e],
                                          bins=self.bin_edges)
            aux_mean_nue, _ = np.histogram(array[self.reco_energy][nu_e],
                                          weights=array['weight'][nu_e],
                                          bins=self.bin_edges)

            self.expected_bin_contents['bg'] += aux_mean_bg
            self.expected_bin_contents['signal'] += aux_mean_nue
        self.total_bg_prediction = self.expected_bin_contents['bg']

    def poissonLogLikelihood(self, mu, obs_bin_contents, pot_scale_factor, external_prediction=None):
        if external_prediction is None:
            prediction = pot_scale_factor*(self.total_bg_prediction + mu*self.expected_bin_contents['signal'])
        else:
            prediction = pot_scale_factor*(external_prediction['bg'] + mu*external_prediction['signal'])

        likelihood_bin_by_bin = -prediction + np.where(prediction!=0, obs_bin_contents*np.log(prediction), 0)
        return likelihood_bin_by_bin.sum(axis=-1)

    def gaussianLogLikelihood(self, mu, obs_bin_contents, pot_scale_factor, external_prediction=None):
        if external_prediction is None:
            prediction = pot_scale_factor*(self.total_bg_prediction + mu*self.expected_bin_contents['signal'])
        else:
            prediction = pot_scale_factor*(external_prediction['bg'] + mu*external_prediction['signal'])

        likelihood_bin_by_bin = np.where(prediction!=0, -0.5*(obs_bin_contents-prediction)**2/prediction - np.log(np.sqrt(prediction)), 0)
        return likelihood_bin_by_bin.sum(axis=-1)

    def gaussianLogLikelihoodApprox(self, mu, obs_bin_contents, pot_scale_factor, external_prediction=None):
        if external_prediction is None:
            prediction = pot_scale_factor*(self.total_bg_prediction + mu*self.expected_bin_contents['signal'])
        else:
            prediction = pot_scale_factor*(external_prediction['bg'] + mu*external_prediction['signal'])

        likelihood_bin_by_bin = np.where(prediction!=0, -0.5*(obs_bin_contents-prediction)**2/prediction, 0)
        return likelihood_bin_by_bin.sum(axis=-1)

    def effectiveLogLikelihood(self, mu, obs_bin_contents, pot_scale_factor, external_prediction=None):
        assert external_prediction is None
        prediction_mean = pot_scale_factor*(self.total_bg_prediction + mu*self.expected_bin_contents['signal'])
        prediction_sigma2 = (self.total_bg_sigma2 + mu**2*self.signal_sigma2)*pot_scale_factor**2
        # import pdb; pdb.set_trace();
        # likelihood_tot = 0
        # for obs, mean, sigma2 in zip(obs_bin_contents, prediction_mean, prediction_sigma2):
        #     likelihood_tot += LEff(obs, mean, sigma2)
        # return likelihood_tot

        likelihood_bin_by_bin = LEff_v(obs_bin_contents, prediction_mean, prediction_sigma2)
        return likelihood_bin_by_bin.sum(axis=-1)

    def neymanPearsonLikelihoodRatio(self, mu_0, mu_1, obs_bin_contents, pot_scale_factor, external_prediction=None):
        return -2*(self.poissonLogLikelihood(mu_0, obs_bin_contents, pot_scale_factor, external_prediction) - self.poissonLogLikelihood(mu_1, obs_bin_contents, pot_scale_factor, external_prediction))

    def gaussianLikelihoodRatio(self, mu_0, mu_1, obs_bin_contents, pot_scale_factor, external_prediction=None):
        return -2*(self.gaussianLogLikelihood(mu_0, obs_bin_contents, pot_scale_factor, external_prediction) - self.gaussianLogLikelihood(mu_1, obs_bin_contents, pot_scale_factor, external_prediction))

    def deltaChi2(self, mu_0, mu_1, obs_bin_contents, pot_scale_factor, external_prediction=None):
        return -2*(self.gaussianLogLikelihoodApprox(mu_0, obs_bin_contents, pot_scale_factor, external_prediction) - self.gaussianLogLikelihoodApprox(mu_1, obs_bin_contents, pot_scale_factor, external_prediction))

    def Chi2_mu0(self, mu_0, mu_1, obs_bin_contents, pot_scale_factor, external_prediction=None):
        return -2*(self.gaussianLogLikelihoodApprox(mu_0, obs_bin_contents, pot_scale_factor, external_prediction))

    def Chi2_mu1(self, mu_0, mu_1, obs_bin_contents, pot_scale_factor, external_prediction=None):
        return -2*(self.gaussianLogLikelihoodApprox(mu_1, obs_bin_contents, pot_scale_factor, external_prediction))

    def logRatioEffectiveLikelihood(self, mu_0, mu_1, obs_bin_contents, pot_scale_factor, external_prediction=None):
        return -2*(self.effectiveLogLikelihood(mu_0, obs_bin_contents, pot_scale_factor, external_prediction) - self.effectiveLogLikelihood(mu_1, obs_bin_contents, pot_scale_factor, external_prediction))

    def pseudoExperiments(self, n_toy, systematic_uncertainties={}, pot_scale_factor=1):
        bg_toy = np.zeros((n_toy, self.n_bin))
        for name, prediction in self.expected_bin_contents.items():
            if name not in systematic_uncertainties.keys():
                systematic_uncertainties[name] = 0
            mean_toy = np.stack([np.random.normal(m, s, n_toy) for m,s in zip(prediction, prediction*systematic_uncertainties[name])], axis=1)
            poisson_toy = np.stack([np.random.poisson(pot_scale_factor*m) for m in mean_toy.clip(min=0)], axis=0)
            if name == 'signal':
                signal_toy = poisson_toy
            else:
                bg_toy += poisson_toy

        return bg_toy, signal_toy

    def pseudoExperimentsFactory(self, n_toy, systematic_uncertainties={}, pot_scale_factors=[1]):
        bg_toy = [np.zeros((n_toy, self.n_bin)) for pot_scale_factor in pot_scale_factors]
        signal_toy = []
        for name, prediction in self.expected_bin_contents.items():
            if name not in systematic_uncertainties.keys():
                systematic_uncertainties[name] = 0
            mean_toy = np.stack([np.random.normal(m, s, n_toy) for m,s in zip(prediction, prediction*systematic_uncertainties[name])], axis=1)
            for i, pot_scale_factor in enumerate(pot_scale_factors):
                poisson_toy = np.stack([np.random.poisson(pot_scale_factor*m) for m in mean_toy.clip(min=0)], axis=0)
                if name == 'signal':
                    signal_toy.append(poisson_toy)
                else:
                    bg_toy[i] += poisson_toy
        return bg_toy, signal_toy

    def pseudoExperimentsFactoryExp(self, n_toy, systematic_uncertainties={}, pot_scale_factors=[1]):
        bg_toy_means = np.zeros((n_toy, self.n_bin))
        bg_toy = []
        signal_toy = []
        for name, prediction in self.expected_bin_contents.items():
            if name not in systematic_uncertainties.keys():
                systematic_uncertainties[name] = 0
            mean_toy = np.stack([np.random.normal(m, s, n_toy) for m,s in zip(prediction, prediction*systematic_uncertainties[name])], axis=1)

            if name == 'signal':
                signal_toy_mean = mean_toy
            else:
                bg_toy_means += mean_toy
        for i, pot_scale_factor in enumerate(pot_scale_factors):
            signal_toy.append(np.stack([np.random.poisson(pot_scale_factor*m) for m in signal_toy_mean.clip(min=0)], axis=0))
            bg_toy.append(np.stack([np.random.poisson(pot_scale_factor*m) for m in bg_toy_means.clip(min=0)], axis=0))
        return bg_toy, signal_toy

    def pvalue2sigma(self, pvalue):
        return mpmath.sqrt(2) * mpmath.erfinv(1 - 2*pvalue)

    def significanceCalculation(self, test_stat_mu0, test_stat_mu1, percentage_values=[16, 50, 84]):
        expected_quantiles = np.percentile(test_stat_mu1, percentage_values)
        expected_pvalues = []
        expected_one_minus_pvalues = []
        expected_significance = []

        for percentage_value, quantile in zip(percentage_values, expected_quantiles):
            one_minus_pvalue = np.less(test_stat_mu0, quantile).sum()/len(test_stat_mu0)
            pvalue = 1. - one_minus_pvalue
            if pvalue != 0:
                significance = float(self.pvalue2sigma(pvalue))
            if pvalue == 0:
                mean = np.mean(test_stat_mu0)
                std = np.std(test_stat_mu0)
                significance = (quantile - mean)/std

            expected_one_minus_pvalues.append(one_minus_pvalue)
            expected_pvalues.append(pvalue)
            expected_significance.append(significance)
        return expected_significance, expected_pvalues, expected_quantiles

    def asimovPlot(self, mu, mu_0, mu_1, pot_scale_factor=1, title=None, ntoy=1000, external_prediction=None, test_stat='pois_llr'):
        assert self.bin_edges is not None
        bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1])/2
        bin_widths = (self.bin_edges[1:] - self.bin_edges[:-1])
        if external_prediction is None:
            bg_prediction = self.total_bg_prediction
            signal_prediction = self.expected_bin_contents['signal']
        else:
            bg_prediction = external_prediction['bg']
            signal_prediction = external_prediction['signal']

        #plot two hypotheses
        plt.bar(bin_centers,
                pot_scale_factor*(bg_prediction + mu_0*signal_prediction),
                width=bin_widths,
                label=r'$H_0$: $\mu$ = {}'.format(mu_0),
                )
        plt.bar(bin_centers,
                pot_scale_factor*(mu_1-mu_0)*signal_prediction,
                bottom=pot_scale_factor*(bg_prediction + mu_0*signal_prediction),
                width=bin_widths,
                label=r'$H_1$: $\mu$ = {}'.format(mu_1),
                )
        # plot pseudo data
        bg_toy, signal_toy = self.pseudoExperiments(n_toy=ntoy, pot_scale_factor=pot_scale_factor)
        toy_mu0 = bg_toy + mu_0 * signal_toy
        # toy_mu1 = bg_toy + mu_1 * signal_toy
        test_stat_mu0 = self.ts_dict[test_stat](mu_0,
                                                          mu_1,
                                                          toy_mu0,
                                                          pot_scale_factor,
                                                          external_prediction)
        # test_stat_mu1 = self.neymanPearsonLikelihoodRatio(mu_0,
        #                                                   mu_1,
        #                                                   toy_mu1,
        #                                                   pot_scale_factor,
        #                                                   external_prediction)
        test_stat_mu1 = self.ts_dict[test_stat](mu_0,
                                                          mu_1,
                                                          pot_scale_factor*(self.total_bg_prediction+self.expected_bin_contents['signal']),
                                                          pot_scale_factor,
                                                          external_prediction)

        expected_significance, expected_pvalues, expected_quantiles = self.significanceCalculation(test_stat_mu0, test_stat_mu1, percentage_values=[50])
        pseudo_data = pot_scale_factor*(self.total_bg_prediction + mu_1*self.expected_bin_contents['signal'])
        plt.plot(bin_centers,
                 pseudo_data,
                 'k.',
                 label=r'Asimov dataset $\mu$ = {}'.format(mu) + '\n' + r'signifcance = {:.2f} $\sigma$'.format(expected_significance[0]),
                 )


        plt.legend()
        plt.ylabel("Expected number of entries")
        plt.xlabel("Reconstructed energy [GeV]")
        if title is None:
            title = self.selection_label + '\n' + self.label
        plt.title(title+"\nMicroBooNE Preliminary - {:.2g} POT".format(pot_scale_factor*self.pot), loc='left')

    def testStatisticsPlot(self, mu_0, mu_1, n_toy=100000, percentage_values=[16, 50, 84], pot_scale_factor=1, systematic_uncertainties={}, n_bins=50, log=False, title=None, external_prediction=None, test_stat='pois_llr', range=None, print_numbers=True):
        bg_toy, signal_toy = self.pseudoExperiments(n_toy, systematic_uncertainties, pot_scale_factor)
        toy_mu0 = bg_toy + mu_0 * signal_toy
        toy_mu1 = bg_toy + mu_1 * signal_toy
        test_stat_mu0 = self.ts_dict[test_stat](mu_0, mu_1, toy_mu0, pot_scale_factor, external_prediction)
        test_stat_mu1 = self.ts_dict[test_stat](mu_0, mu_1, toy_mu1, pot_scale_factor, external_prediction)
        expected_significance, expected_pvalues, expected_quantiles = self.significanceCalculation(test_stat_mu0, test_stat_mu1, percentage_values)

        bin_contents_total, bin_edges, _ = plt.hist(
                                             [test_stat_mu0, test_stat_mu1],
                                             bins=n_bins,
                                             range=range,
                                             density=True,
                                             label=[r'$\mu$ = {:.2g}'.format(mu_0), r'$\mu$ = {:.2g}'.format(mu_1)],
                                             alpha=0.7,
                                             histtype='stepfilled',
                                             lw=2,
                                             log=log,
                                             )

        bin_width = bin_edges[1] - bin_edges[0]
        plt.ylabel("Probability / {:.2f}".format(bin_width))
        plt.xlabel(self.ts_labels[test_stat])
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        heights = {16: 0.5, 50: 0.7, 84: 0.5}
        horizontalalignments = {16: 'right', 50: 'center', 84: 'left'}
        position_offset = {16: -5, 50: 0, 84: +5}

        for i, percentage_value in enumerate(percentage_values):
            quantile = expected_quantiles[i]
            pvalue = expected_pvalues[i]
            significance = expected_significance[i]

            plt.axvline(quantile, ymax=heights[percentage_value]-0.1, color='red', linestyle='--', label='expected {}%'.
                        format(percentage_value))
            if print_numbers:
                plt.text(quantile+position_offset[percentage_value],
                         heights[percentage_value]*ymax,
                         'p = {:.1e}\nZ = {:.2f}'.format(pvalue, significance)+r'$\sigma$',
                         fontsize=10,
                         verticalalignment='center',
                         horizontalalignment=horizontalalignments[percentage_value],
                         )

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.04,1), loc="best")

        textstr_list = [r'Events $\pm$ systematic']
        for name, prediction in self.expected_bin_contents.items():
            if name not in systematic_uncertainties.keys():
                systematic_uncertainties[name] = 0
            textstr_list.append(r'{}: {:.2g} $\pm$ {:.2g}%'.format(name, pot_scale_factor*prediction.sum(), systematic_uncertainties[name]*100))
        textstr = '\n'.join(textstr_list)
        props = dict(boxstyle='round', facecolor='white', alpha=0.5, linewidth=0.5)
        plt.text(1.08, 0.5, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        if title is None:
            title = self.selection_label + '\n' + self.label
        plt.title(title+"\nMicroBooNE Preliminary - {:.2g} POT".format(pot_scale_factor*self.pot), loc='left')
        plt.tight_layout()
        return expected_significance

    def SignificanceFunctionScaleFactors(self, mu_0, mu_1, n_toy=100000, percentage_values=[16, 50, 84], pot_scale_factors=[1], systematic_uncertainties={}, label='', title='', type='discovery', test_stat='pois_llr'):
        expectation = []
        bg_toy, signal_toy = self.pseudoExperimentsFactoryExp(n_toy, systematic_uncertainties, pot_scale_factors)

        for i, pot_scale_factor in enumerate(pot_scale_factors):
            toy_mu0 = bg_toy[i] + mu_0*signal_toy[i]
            toy_mu1 = bg_toy[i] + mu_1*signal_toy[i]
            test_stat_mu0 = self.ts_dict[test_stat](mu_0, mu_1, toy_mu0, pot_scale_factor)
            test_stat_mu1 = self.ts_dict[test_stat](mu_0, mu_1, toy_mu1, pot_scale_factor)
            aux_significance, aux_pvalues, aux_quantiles = self.significanceCalculation(test_stat_mu0, test_stat_mu1, percentage_values)
            if type == 'discovery':
                expectation.append(aux_significance)
            elif type == 'exclusion':
                expectation.append(aux_pvalues)

        expectation = np.array(expectation)
        print(expectation)
        if type == 'exclusion':
            expectation = 100*(1.-expectation)

        x_axis_labels = self.pot*np.array(pot_scale_factors)
        plt.plot(x_axis_labels, expectation[:, 1], label=label)
        plt.fill_between(x_axis_labels, expectation[:, 0], expectation[:, 2],
            alpha=0.2,
            linewidth=0, antialiased=True, label='expected {:.2g}%'.format(percentage_values[2] - percentage_values[0]))
        plt.legend()
        if type == 'discovery':
            plt.ylabel(r'Expected significance [$\sigma$]')
        elif type == 'exclusion':
            plt.ylabel(r'Esclusion Confidence Level [$\%$]')
        plt.xlabel('Collected POT')
        plt.title(title, loc='left')
        plt.title(title+"\nMicroBooNE Preliminary", loc='left')
        plt.tight_layout()

    def SignificanceFunctionSystematicUncertainties(self, mu_0, mu_1, n_toy=100000, percentage_values=[16, 50, 84], systematic_uncertainties=[0], pot_scale_factor=1, title='', type='discovery', test_stat='pois_llr'):
        expectation = []
        for systematic in systematic_uncertainties:
            syst_dict = dict(zip(self.expected_bin_contents.keys(), [systematic]*len(self.expected_bin_contents.keys())))
            bg_toy, signal_toy = self.pseudoExperimentsFactoryExp(n_toy, syst_dict, [pot_scale_factor])
            toy_mu0 = bg_toy[0] + mu_0*signal_toy[0]
            toy_mu1 = bg_toy[0] + mu_1*signal_toy[0]
            test_stat_mu0 = self.ts_dict[test_stat](mu_0, mu_1, toy_mu0, pot_scale_factor)
            test_stat_mu1 = self.ts_dict[test_stat](mu_0, mu_1, toy_mu1, pot_scale_factor)
            aux_significance, aux_pvalues, aux_quantiles = self.significanceCalculation(test_stat_mu0, test_stat_mu1, percentage_values)
            if type == 'discovery':
                expectation.append(aux_significance)
            elif type == 'exclusion':
                expectation.append(aux_pvalues)

        expectation = np.array(expectation)
        if type == 'exclusion':
            expectation = 100*(1.-expectation)

        x_axis_labels = np.array(systematic_uncertainties)*100
        plt.plot(x_axis_labels, expectation[:, 1], label='{:.2g} POT'.format(pot_scale_factor*self.pot))
        plt.fill_between(x_axis_labels, expectation[:, 0], expectation[:, 2],
            alpha=0.2,
            linewidth=0, antialiased=True, label='expected {:.2g}%'.format(percentage_values[2] - percentage_values[0]))
        plt.legend()
        if type == 'discovery':
            plt.ylabel(r'Expected significance [$\sigma$]')
        elif type == 'exclusion':
            plt.ylabel(r'Esclusion Confidence Level [$\%$]')
        plt.xlabel(r'Systematic Uncertainty [$\%$]')
        plt.title(title, loc='left')
        plt.title(title+"\nMicroBooNE Preliminary", loc='left')
        plt.tight_layout()
