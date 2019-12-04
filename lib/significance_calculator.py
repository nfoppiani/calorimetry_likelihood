import numpy as np
from matplotlib import pyplot as plt
import mpmath

class significanceCalculator(object):

    def __init__(self, expected_bin_contents, bin_edges=None, pot=1):
        assert 'signal' in expected_bin_contents.keys()
        self.expected_bin_contents = expected_bin_contents
        self.bin_edges = bin_edges
        self.n_bin = len(expected_bin_contents['signal'])
        self.total_bg_prediction = np.zeros(self.n_bin)
        for name, prediction in expected_bin_contents.items():
            if name == 'signal':
                continue
            self.total_bg_prediction += prediction

        self.pot = pot

    def poissonLogLikelihood(self, mu, obs_bin_contents, pot_scale_factor, external_prediction=None):
        if external_prediction is None:
            prediction = pot_scale_factor*(self.total_bg_prediction + mu*self.expected_bin_contents['signal'])
        else:
            prediction = pot_scale_factor*(external_prediction['bg'] + mu*external_prediction['signal'])

        likelihood_bin_by_bin = -prediction + np.where(prediction!=0, obs_bin_contents*np.log(prediction), 0)
        return likelihood_bin_by_bin.sum(axis=-1)

    def neymanPearsonLikelihoodRatio(self, mu_0, mu_1, obs_bin_contents, pot_scale_factor, external_prediction=None):
        return -2*(self.poissonLogLikelihood(mu_0, obs_bin_contents, pot_scale_factor, external_prediction) - self.poissonLogLikelihood(mu_1, obs_bin_contents, pot_scale_factor, external_prediction))

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
            significance = float(self.pvalue2sigma(pvalue))
            expected_one_minus_pvalues.append(one_minus_pvalue)
            expected_pvalues.append(pvalue)
            expected_significance.append(significance)
        return expected_significance, expected_pvalues, expected_quantiles

    def asimovPlot(self, mu, mu_0, mu_1, pot_scale_factor=1, title='', ntoy=1000, external_prediction=None):
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
                pot_scale_factor*(bg_prediction + mu_1*signal_prediction),
                width=bin_widths,
                label=r'$H_1$: $\mu$ = {}'.format(mu_1),
                )
        plt.bar(bin_centers,
                pot_scale_factor*(bg_prediction + mu_0*signal_prediction),
                width=bin_widths,
                label=r'$H_0$: $\mu$ = {}'.format(mu_0),
                )
        # plot pseudo data
        bg_toy, signal_toy = self.pseudoExperiments(n_toy=ntoy, pot_scale_factor=pot_scale_factor)
        toy_mu0 = bg_toy + mu_0 * signal_toy
        # toy_mu1 = bg_toy + mu_1 * signal_toy
        test_stat_mu0 = self.neymanPearsonLikelihoodRatio(mu_0,
                                                          mu_1,
                                                          toy_mu0,
                                                          pot_scale_factor,
                                                          external_prediction)
        # test_stat_mu1 = self.neymanPearsonLikelihoodRatio(mu_0,
        #                                                   mu_1,
        #                                                   toy_mu1,
        #                                                   pot_scale_factor,
        #                                                   external_prediction)
        test_stat_mu1 = self.neymanPearsonLikelihoodRatio(mu_0,
                                                          mu_1,
                                                          pot_scale_factor*(self.total_bg_prediction+self.expected_bin_contents['signal']),
                                                          pot_scale_factor,
                                                          external_prediction)

        expected_significance, expected_pvalues, expected_quantiles = self.significanceCalculation(test_stat_mu0, test_stat_mu1, percentage_values=[50])
        pseudo_data = pot_scale_factor*(self.total_bg_prediction + mu*self.expected_bin_contents['signal'])
        plt.plot(bin_centers,
                 pseudo_data,
                 'k.',
                 label=r'Asimov dataset $\mu$ = {}'.format(mu) + '\n' + r'signifcance = {:.2f} $\sigma$'.format(expected_significance[0]),
                 )


        plt.legend()
        plt.ylabel("Expected number of entries")
        plt.xlabel("Reconstructed energy [GeV]")
        plt.title(title+"\nMicroBooNE Preliminary - {:.2g} POT".format(pot_scale_factor*self.pot), loc='left')

    def testStatisticsPlot(self, mu_0, mu_1, n_toy=100000, percentage_values=[16, 50, 84], pot_scale_factor=1, systematic_uncertainties={}, n_bins=50, log=False, title='', external_prediction=None):
        bg_toy, signal_toy = self.pseudoExperiments(n_toy, systematic_uncertainties, pot_scale_factor)
        toy_mu0 = bg_toy + mu_0 * signal_toy
        toy_mu1 = bg_toy + mu_1 * signal_toy
        test_stat_mu0 = self.neymanPearsonLikelihoodRatio(mu_0, mu_1, toy_mu0, pot_scale_factor, external_prediction)
        test_stat_mu1 = self.neymanPearsonLikelihoodRatio(mu_0, mu_1, toy_mu1, pot_scale_factor, external_prediction)

        expected_significance, expected_pvalues, expected_quantiles = self.significanceCalculation(test_stat_mu0, test_stat_mu1, percentage_values)

        bin_contents_total, bin_edges, _ = plt.hist(
                                             [test_stat_mu0, test_stat_mu1],
                                             bins=n_bins,
                                             density=True,
                                             label=[r'$\mu$ = {:.2g}'.format(mu_0), r'$\mu$ = {:.2g}'.format(mu_1)],
                                             alpha=0.7,
                                             histtype='stepfilled',
                                             lw=2,
                                             log=log,
                                             )

        bin_width = bin_edges[1] - bin_edges[0]
        plt.ylabel("Probability / {:.2f}".format(bin_width))
        plt.xlabel("Neyman Pearson likelihood ratio")
        plt.title(title+"\nMicroBooNE Preliminary - {:.2g} POT".format(pot_scale_factor*self.pot), loc='left')
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
        plt.tight_layout()
        return expected_significance

    def SignificanceFunctionScaleFactors(self, mu_0, mu_1, n_toy=100000, percentage_values=[16, 50, 84], pot_scale_factors=[1], systematic_uncertainties={}, label='', title='', type='discovery'):
        expectation = []
        bg_toy, signal_toy = self.pseudoExperimentsFactoryExp(n_toy, systematic_uncertainties, pot_scale_factors)

        for i, pot_scale_factor in enumerate(pot_scale_factors):
            toy_mu0 = bg_toy[i] + mu_0*signal_toy[i]
            toy_mu1 = bg_toy[i] + mu_1*signal_toy[i]
            test_stat_mu0 = self.neymanPearsonLikelihoodRatio(mu_0, mu_1, toy_mu0, pot_scale_factor)
            test_stat_mu1 = self.neymanPearsonLikelihoodRatio(mu_0, mu_1, toy_mu1, pot_scale_factor)
            aux_significance, aux_pvalues, aux_quantiles = self.significanceCalculation(test_stat_mu0, test_stat_mu1, percentage_values)
            if type == 'discovery':
                expectation.append(aux_significance)
            elif type == 'exclusion':
                expectation.append(aux_pvalues)

        expectation = np.array(expectation)
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

    def SignificanceFunctionSystematicUncertainties(self, mu_0, mu_1, n_toy=100000, percentage_values=[16, 50, 84], systematic_uncertainties=[0], pot_scale_factor=1, title='', type='discovery'):
        expectation = []
        for systematic in systematic_uncertainties:
            syst_dict = dict(zip(self.expected_bin_contents.keys(), [systematic]*len(self.expected_bin_contents.keys())))
            bg_toy, signal_toy = self.pseudoExperimentsFactoryExp(n_toy, syst_dict, [pot_scale_factor])
            toy_mu0 = bg_toy[0] + mu_0*signal_toy[0]
            toy_mu1 = bg_toy[0] + mu_1*signal_toy[0]
            test_stat_mu0 = self.neymanPearsonLikelihoodRatio(mu_0, mu_1, toy_mu0, pot_scale_factor)
            test_stat_mu1 = self.neymanPearsonLikelihoodRatio(mu_0, mu_1, toy_mu1, pot_scale_factor)
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
