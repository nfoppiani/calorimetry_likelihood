import numpy as np
from matplotlib import pyplot as plt
import mpmath

class significanceCalculator(object):

    def __init__(self, expected_bin_contents, pot=1):
        assert 'signal' in expected_bin_contents.keys()
        self.expected_bin_contents = expected_bin_contents
        self.n_bin = len(expected_bin_contents['signal'])
        self.total_bg_prediction = np.zeros(self.n_bin)
        for name, prediction in expected_bin_contents.items():
            if name == 'signal':
                continue
            self.total_bg_prediction += prediction

        self.pot = pot

    def poissonLogLikelihood(self, mu, obs_bin_contents):
        prediction = (self.total_bg_prediction + mu*self.expected_bin_contents['signal'])

        likelihood_bin_by_bin = -prediction + np.where(prediction!=0, obs_bin_contents*np.log(prediction), 0)
        return likelihood_bin_by_bin.sum(axis=1)

    def neymanPearsonLikelihoodRatio(self, mu_0, mu_1, obs_bin_contents):
        return -2*(self.poissonLogLikelihood(mu_0, obs_bin_contents) - self.poissonLogLikelihood(mu_1, obs_bin_contents))

    def pseudoExperiments(self, n_toy, mu, systematic_uncertainties={}, scaling_factor=1):
        out_toy = np.zeros((n_toy, self.n_bin))
        mean_toy = {}
        for name, prediction in self.expected_bin_contents.items():
            if name not in systematic_uncertainties.keys():
                systematic_uncertainties[name] = 0
            mean_toy = np.stack([np.random.normal(m, s, n_toy) for m,s in zip(prediction, prediction*systematic_uncertainties[name])], axis=1)
            poisson_toy = np.stack([np.random.poisson(scaling_factor*m) for m in mean_toy.clip(min=0)], axis=0)
            if name == 'signal':
                out_toy += poisson_toy * mu
            else:
                out_toy += poisson_toy

        return out_toy

    def pseudoExperimentsFactory(self, n_toy, systematic_uncertainties={}, scale_factors=[1]):
        bg_toy = [np.zeros((n_toy, self.n_bin)) for scale_factor in scale_factors]
        signal_toy = []
        for name, prediction in self.expected_bin_contents.items():
            if name not in systematic_uncertainties.keys():
                systematic_uncertainties[name] = 0
            mean_toy = np.stack([np.random.normal(m, s, n_toy) for m,s in zip(prediction, prediction*systematic_uncertainties[name])], axis=1)
            for i, scale_factor in enumerate(scale_factors):
                poisson_toy = np.stack([np.random.poisson(scale_factor*m) for m in mean_toy.clip(min=0)], axis=0)
                if name == 'signal':
                    signal_toy.append(poisson_toy)
                else:
                    bg_toy[i] += poisson_toy
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

    def testStatisticsPlot(self, mu_0, mu_1, n_toy_0=100000, n_toy_1=1000, percentage_values=[16, 50, 84], scaling_factor=1, systematic_uncertainties={}, n_bins=50, log=False):
        toy_mu0 = self.pseudoExperiments(n_toy_0, mu_0, systematic_uncertainties, scaling_factor)
        toy_mu1 = self.pseudoExperiments(n_toy_1, mu_1, systematic_uncertainties, scaling_factor)
        test_stat_mu0 = self.neymanPearsonLikelihoodRatio(mu_0, mu_1, toy_mu0)
        test_stat_mu1 = self.neymanPearsonLikelihoodRatio(mu_0, mu_1, toy_mu1)

        expected_significance, expected_pvalues, expected_quantiles = self.significanceCalculation(test_stat_mu0, test_stat_mu1, percentage_values)

        bin_contents_total, bin_edges, _ = plt.hist(
                                             [test_stat_mu1, test_stat_mu0],
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
        plt.title("MicroBooNE Preliminary", loc='right')
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
            textstr_list.append(r'{}: {:.2g} $\pm$ {:.2g}%'.format(name, prediction.sum(), systematic_uncertainties[name]*100))
        textstr = '\n'.join(textstr_list)
        props = dict(boxstyle='round', facecolor='white', alpha=0.5, linewidth=0.5)
        plt.text(1.08, 0.5, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

        return expected_significance

    def SignificanceFunctionScaleFactors(self, mu_0, mu_1, n_toy=100000, percentage_values=[16, 50, 84], scale_factors=[1], systematic_uncertainties={}, label='', title=''):
        expected_significance = []
        bg_toy, signal_toy = self.pseudoExperimentsFactory(n_toy, systematic_uncertainties, scale_factors)

        for i, scale_factor in enumerate(scale_factors):
            toy_mu0 = bg_toy[i] + mu_0*signal_toy[i]
            toy_mu1 = bg_toy[i] + mu_1*signal_toy[i]
            test_stat_mu0 = self.neymanPearsonLikelihoodRatio(mu_0, mu_1, toy_mu0)
            test_stat_mu1 = self.neymanPearsonLikelihoodRatio(mu_0, mu_1, toy_mu1)
            aux_significance, aux_pvalues, aux_quantiles = self.significanceCalculation(test_stat_mu0, test_stat_mu1, percentage_values)
            expected_significance.append(aux_significance)

        expected_significance = np.array(expected_significance)

        x_axis_labels = self.pot*np.array(scale_factors)
        plt.plot(x_axis_labels, expected_significance[:, 1], label=label)
        plt.fill_between(x_axis_labels, expected_significance[:, 0], expected_significance[:, 2],
            alpha=0.2,
            linewidth=0, antialiased=True, label='expected {:.2g}%'.format(percentage_values[2] - percentage_values[0]))

        plt.legend()
        plt.ylabel(r'Expected significance [$\sigma$]')
        plt.xlabel('Collected POT')
        plt.title(title, loc='left')
        plt.title('MicroBooNE preliminary', loc='right')
