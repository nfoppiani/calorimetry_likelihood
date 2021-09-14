import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm

import awkward

def chunked2jagged(array):
    return awkward.concatenate(array.chunks)

class plotter():

    def __init__(self, arrays, scale_factors, pot_beam_on, branch_weights_name=None):
        self.arrays = arrays
        self.scale_factors = scale_factors
        self.pot_beam_on = pot_beam_on

        for name, array in arrays.items():
            if (branch_weights_name is None) or ('beam' in name):
                array['weight'] = np.ones(len(array))
            else:
                array['weight'] = array[branch_weights_name]
            array['weight'] = array['weight']*self.scale_factors[name]

    def plot_pot_normalised_var(self, variable, binning, categories, function='flatten', additional_selection=None, prediction_datasets=['bnb_dirt', 'bnb_nu', 'bnb_nue'], title=None, xlabel=None, log=False, subtracted=False, onebin=False):
        fig, ax = plt.subplots(ncols=1,
                               nrows=2,
                               figsize=(5.1*1.6,5),
                               sharex='col',
                               gridspec_kw={'height_ratios':[3, 1]})

        # create additional selection masks
        additional_selection_masks = {}
        for name, array in self.arrays.items():
            if additional_selection is not None:
                selection_mask = additional_selection(array)
            else:
                selection_mask = (array[variable] == array[variable])
            additional_selection_masks[name] = selection_mask

        # prediction
        labels_predictions = []
        datasets_predictions = []
        weights_predictions = []
        colors = []
        if subtracted is False:
            # do beam OFF
            dataset_name = 'beam_off'
            colors.append('C0')
            array = self.arrays[dataset_name]
            label_prediction = 'DATA Beam OFF'
            labels_predictions.append(label_prediction)

            additional_selection_mask = chunked2jagged(additional_selection_masks[dataset_name])
            array_variable = chunked2jagged(array[variable])

            if function == 'flatten':
                partial_prediction = array_variable[additional_selection_mask]
            else:
                partial_prediction = getattr(array_variable[additional_selection_mask], function)()

            element_mask = (partial_prediction == array_variable[additional_selection_mask])
            datasets_prediction = array_variable[additional_selection_mask][element_mask].flatten()

            extended_weights = (array['weight'] * (array_variable == array_variable))
            weights_prediction = extended_weights[additional_selection_mask][element_mask].flatten()

            datasets_predictions.append(datasets_prediction)
            weights_predictions.append(weights_prediction)

            del additional_selection_mask, array_variable, partial_prediction, element_mask, extended_weights
            gc.collect()

            # do beam ON
            dataset_name = 'beam_on'
            label_data = 'DATA Beam ON'
            array = self.arrays[dataset_name]
            array_variable = chunked2jagged(array[variable])
            additional_selection_mask = chunked2jagged(additional_selection_masks[dataset_name])
            datasets_data = getattr(array_variable[additional_selection_mask], function)()
            beam_on_values, bin_edges = np.histogram(datasets_data,
                                             bins=binning[0],
                                             range=(binning[1], binning[2])
                                           )
            beam_on_y_err = np.sqrt(beam_on_values)
            del additional_selection_mask, array_variable, datasets_data
            gc.collect()
        else:
            dataset_name = 'beam_on'
            array = self.arrays[dataset_name]
            array_variable = chunked2jagged(array[variable])
            additional_selection_mask = chunked2jagged(additional_selection_masks[dataset_name])
            beam_on_data = getattr(array_variable[additional_selection_mask], function)()
            beam_on_aux, bin_edges = np.histogram(beam_on_data,
                                             bins=binning[0],
                                             range=(binning[1], binning[2])
                                           )
            del additional_selection_mask, array_variable, beam_on_data
            gc.collect()

            dataset_name = 'beam_on'
            array = self.arrays[dataset_name]
            array_variable = chunked2jagged(array[variable])
            additional_selection_mask = chunked2jagged(additional_selection_masks[dataset_name])
            beam_off_data = getattr(array_variable[additional_selection_mask], function)()
            beam_off_aux, bin_edges = np.histogram(beam_off_data,
                                             bins=binning[0],
                                             range=(binning[1], binning[2]),
                                           )

            label_data = 'DATA Beam ON - Beam OFF'
            beam_on_values = beam_on_aux - beam_off_aux * self.scale_factors['beam_off']
            beam_on_y_err = np.sqrt(beam_on_aux + beam_off_aux * self.scale_factors['beam_off']**2)
            del additional_selection_mask, array_variable, beam_off_data
            gc.collect()

        aux_datasets_predictions = {}
        aux_weights_predictions = {}

        for dataset_name in prediction_datasets:
            array = self.arrays[dataset_name]
            array_variable = chunked2jagged(array[variable])
            additional_selection_mask = chunked2jagged(additional_selection_masks[dataset_name])

            if function == 'flatten':
                partial_prediction = array_variable[additional_selection_mask]
            else:
                partial_prediction = getattr(array_variable[additional_selection_mask], function)()

            element_mask = (partial_prediction == array_variable[additional_selection_mask])

            for (category_branch_name, category_label) in categories.items():
                extended_category_mask = chunked2jagged(array[category_branch_name] * (array_variable == array_variable))
                category_mask = extended_category_mask[additional_selection_mask]
                full_mask = (category_mask & element_mask)
                prediction = array_variable[additional_selection_mask][full_mask].flatten()
                extended_weights = (array['weight'] * (array_variable == array_variable))
                weights = extended_weights[additional_selection_mask][full_mask].flatten()

                if category_label not in aux_datasets_predictions.keys():
                    aux_datasets_predictions[category_label] = [prediction]
                    aux_weights_predictions[category_label] = [weights]
                else:
                    aux_datasets_predictions[category_label].append(prediction)
                    aux_weights_predictions[category_label].append(weights)

                del extended_category_mask, category_mask, full_mask, extended_weights
                gc.collect()

            del array_variable, additional_selection_mask, partial_prediction, element_mask
            gc.collect()

        for i, category_label in enumerate(categories.values()):
            labels_predictions.append(category_label)
            colors.append('C{}'.format(i+1))

            datasets_predictions.append(np.concatenate(aux_datasets_predictions[category_label]))
            weights_predictions.append(np.concatenate(aux_weights_predictions[category_label]))

        # for name, dataset, weight in zip(labels_predictions, datasets_predictions, weights_predictions):
        #     print(name, len(dataset), len(weight))

        bin_contents_prediction, bin_edges, patches_hist = ax[0].hist(datasets_predictions,
                 bins=binning[0],
                 range=(binning[1], binning[2]),
                 label=labels_predictions,
                 weights=weights_predictions,
                 stacked=True,
                 color=colors,
                 log=log,
                )

        # prediction errors and plot of patches
        total_datasets = np.concatenate(datasets_predictions)
        total_weights = np.concatenate(weights_predictions)

        bin_indices = np.digitize(total_datasets, bin_edges)

        binned_weights = np.asarray([total_weights[np.where(bin_indices == idx)[0]] for idx in range(1, len(bin_edges))])
        bin_errors_prediction = np.asarray([np.sqrt(np.sum(np.square(w))) for w in binned_weights])

        bin_width = (binning[2] - binning[1])/binning[0]
        for bin_edge, prediction_cont, prediction_err in zip(bin_edges[:-1], bin_contents_prediction[-1], bin_errors_prediction):
            ax[0].add_patch(patches.Rectangle(xy=(bin_edge, prediction_cont-prediction_err),
                                              width=bin_width,
                                              height=prediction_err*2,
                                              hatch="\\\\\\\\\\",
                                              Fill=False,
                                              linewidth=0,alpha=0.4))

        # now plot the data
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
        beam_on_x_err = [(bin_edges[i+1]-bin_edges[i])/2 for i in range(len(bin_edges)-1)]
        ax[0].errorbar(bin_centers,
                         beam_on_values,
                         xerr=beam_on_x_err,
                         yerr=beam_on_y_err,
                         fmt='k.',
                         label=label_data)

        # ratio
        bin_contents_ratio = beam_on_values / bin_contents_prediction[-1]
        bin_relative_errors_prediction = bin_errors_prediction / bin_contents_prediction[-1]
        bin_relative_errors_data = beam_on_y_err / beam_on_values
        relative_bin_errors_ratio = np.sqrt(bin_relative_errors_prediction**2 + bin_relative_errors_data**2)
        bin_errors_ratio = bin_contents_ratio*relative_bin_errors_ratio

        ax[1].plot([bin_edges[0], bin_edges[-1]], np.ones(2), 'r--')
        ax[1].errorbar(bin_centers,
                       bin_contents_ratio,
                       xerr=beam_on_x_err,
                       yerr=bin_errors_ratio,
                       fmt="k.")

        # setting title and labels
        if title is not None:
            title = (title + '\nMicroBooNE Preliminary')
        else:
            title = 'MicroBooNE Preliminary'
        ax[0].set_title(title, loc='left')
        ax[0].set_title('BNB {:.3f}e19 POT'.format(self.pot_beam_on/1e19), loc='right')
        ax[0].autoscale()

        # chi2
        chi2_bins = (beam_on_values - bin_contents_prediction[-1])**2 / (bin_contents_prediction[-1] + bin_errors_prediction**2)
        chi2 = np.sum(chi2_bins[~np.isnan(chi2_bins)])
        ndof = len(chi2_bins[~np.isnan(chi2_bins)])
        ax[0].text(x=0.6, y=0.9,
                   s=r"$X^2$ = {:.1f}, ndf = {:.0f}".format(chi2, ndof),
                   transform=ax[0].transAxes)

        if onebin:
            ax[1].set_xlabel("One bin")
            ax[0].set_ylabel("Entries")
            ax[0].tick_params(axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off
            ax[1].tick_params(axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off
        else:
            if xlabel is None:
                xlabel = variable
            ax[1].set_xlabel(xlabel)
            ax[0].set_ylabel("Entries / {:.2f}".format(bin_width))

        handles, labels = ax[0].get_legend_handles_labels()
        ax[0].legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.04,1), loc="upper left")

        ax[1].set_ylim(0.5, 1.5)
        if subtracted is False:
            ax[1].set_ylabel('DATA/PREDICTION')
        else:
            ax[1].set_ylabel('(ON - OFF)/MC')

        fig.tight_layout(h_pad=0.5)
        return fig

    def plot_one_sample(self, dataset_name, variable, binning, categories, function='flatten', additional_selection=None, title=None, xlabel=None, log=False, onebin=False):
        fig = plt.figure(figsize=(5.1*1.6,5))

        ax = plt.gca()
        array = self.arrays[dataset_name]

        if additional_selection is not None:
            additional_selection_mask = additional_selection(array)
        else:
            additional_selection_mask = (array[variable] == array[variable])

        # prediction
        labels_predictions = []
        datasets_predictions = []
        weights_predictions = []

        for category_branch_name, category_label in categories.items():
            labels_predictions.append(category_label)

            if function == 'flatten':
                partial_prediction = array[variable][additional_selection_mask]
            else:
                partial_prediction = getattr(array[variable][additional_selection_mask], function)()

            element_mask = (partial_prediction == array[variable][additional_selection_mask])
            extended_category_mask = array[category_branch_name] * (array[variable] == array[variable])
            category_mask = extended_category_mask[additional_selection_mask]
            full_mask = (category_mask & element_mask)
            prediction = array[variable][additional_selection_mask][full_mask].flatten()
            extended_weights = (array['weight'] * (array[variable] == array[variable]))
            weights = extended_weights[additional_selection_mask][full_mask].flatten()

            datasets_predictions.append(prediction)
            weights_predictions.append(weights)

        bin_contents_prediction, bin_edges, patches_hist = ax.hist(datasets_predictions,
                 bins=binning[0],
                 range=(binning[1], binning[2]),
                 label=labels_predictions,
                 weights=weights_predictions,
                 stacked=True,
                 log=log,
                )

        # prediction errors and plot of patches
        total_datasets = np.concatenate(datasets_predictions)
        total_weights = np.concatenate(weights_predictions)

        bin_indices = np.digitize(total_datasets, bin_edges)

        binned_weights = np.asarray([total_weights[np.where(bin_indices == idx)[0]] for idx in range(1, len(bin_edges))])
        bin_errors_prediction = np.asarray([np.sqrt(np.sum(np.square(w))) for w in binned_weights])

        bin_width = (binning[2] - binning[1])/binning[0]
        for bin_edge, prediction_cont, prediction_err in zip(bin_edges[:-1], bin_contents_prediction[-1], bin_errors_prediction):
            ax.add_patch(patches.Rectangle(xy=(bin_edge, prediction_cont-prediction_err),
                                              width=bin_width,
                                              height=prediction_err*2,
                                              hatch="\\\\\\\\\\",
                                              Fill=False,
                                              linewidth=0,alpha=0.4))

        # setting titles etc.
        if title is not None:
            title = (title + '\nMicroBooNE Preliminary')
        else:
            title = 'MicroBooNE Preliminary'
        ax.set_title(title, loc='left')
        ax.set_title('BNB {:.3f}e19 POT'.format(self.pot_beam_on/1e19), loc='right')
        ax.autoscale()

        if onebin:
            ax.set_xlabel("One bin")
            ax.set_ylabel("Entries")
            ax.tick_params(axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off
            ax.tick_params(axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off
        else:
            if xlabel is None:
                xlabel = variable
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Entries / {:.2f}".format(bin_width))

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.04,1), loc="upper left")
        plt.tight_layout()

    def plot2d(self, dataset_name, variables, binning, additional_selection=None, title=None, labels=None, log=False):

        plt.figure()
        ax = plt.gca()
        array = self.arrays[dataset_name]

        if additional_selection is not None:
            additional_selection_mask = additional_selection(array)
        else:
            additional_selection_mask = (array[variable] == array[variable])

        partial_prediction_x = array[variables[0]][additional_selection_mask]
        partial_prediction_y = array[variables[1]][additional_selection_mask]

        element_mask = (partial_prediction_x == array[variables[0]][additional_selection_mask])
        prediction_x = partial_prediction_x.flatten()
        prediction_y = partial_prediction_y.flatten()
        extended_weights = (array['weight'] * (array[variables[0]] == array[variables[0]]))
        weights = extended_weights[additional_selection_mask][element_mask].flatten()

        x = array[variables[0]][additional_selection_mask].flatten()
        y = array[variables[1]][additional_selection_mask].flatten()
        if log:
            plt.hist2d(prediction_x, prediction_y,
                       bins=(binning[0], binning[3]),
                       range=( (binning[1], binning[2]), (binning[4], binning[5])),
                       norm=LogNorm(),
                       weights=weights
                       )
        else:
            plt.hist2d(prediction_x, prediction_y,
                       bins=(binning[0], binning[3]),
                       range=( (binning[1], binning[2]), (binning[4], binning[5])),
                       weights=weights
                       )
        plt.colorbar()

        # setting titles etc.
        if title is not None:
            title = (title + '\nMicroBooNE Preliminary')
        else:
            title = 'MicroBooNE Preliminary'
        ax.set_title(title, loc='left')
        ax.autoscale()

        if labels is None:
            xlabel = variables[0]
            ylabel = variables[1]
        else:
            xlabel = labels[0]
            ylabel = labels[1]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()

    def efficiency_one_sample(self, dataset_name, variable, binning, selection_num, selection_den=None, label=None, errors=False, function='flatten', title=None, xlabel=None, log=False, onebin=False):

        ax = plt.gca()
        array = self.arrays[dataset_name]

        num_mask = selection_num(array)
        if selection_den is not None:
            den_mask = selection_den(array)
        else:
            den_mask = (array[variable] == array[variable])

        if function == 'flatten':
            num_partial_prediction = array[variable][num_mask]
            one_minus_num_partial_prediction = array[variable][~num_mask]
            den_partial_prediction = array[variable][den_mask]
        else:
            num_partial_prediction = getattr(array[variable][num_mask], function)()
            one_minus_num_partial_prediction = getattr(array[variable][~num_mask], function)()
            den_partial_prediction = getattr(array[variable][den_mask], function)()

        num_element_mask = (num_partial_prediction == array[variable][num_mask])
        num_prediction = array[variable][num_mask][num_element_mask].flatten()

        one_minus_num_element_mask = (one_minus_num_partial_prediction == array[variable][~num_mask])
        one_minus_num_prediction = array[variable][~num_mask][one_minus_num_element_mask].flatten()

        den_element_mask = (den_partial_prediction == array[variable][den_mask])
        den_prediction = array[variable][den_mask][den_element_mask].flatten()

        extended_weights = (array['weight'] * (array[variable] == array[variable]))
        num_weights = extended_weights[num_mask][num_element_mask].flatten()
        one_minus_num_weights = extended_weights[~num_mask][one_minus_num_element_mask].flatten()
        den_weights = extended_weights[den_mask][den_element_mask].flatten()

        num_content, bin_edges = np.histogram(num_prediction,
                                              bins=binning[0],
                                              range=(binning[1], binning[2]),
                                              weights=num_weights)
        den_content, bin_edges = np.histogram(den_prediction,
                                              bins=binning[0],
                                              range=(binning[1], binning[2]),
                                              weights=den_weights)

        efficiency = num_content/den_content
        print("total efficiency {:.2f}".format(num_content.sum()/den_content.sum()))
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
        bin_width = (binning[2] - binning[1])/binning[0]

        if errors:
            bin_indices_num = np.digitize(num_prediction, bin_edges)
            binned_weights_num = np.asarray([num_weights[np.where(bin_indices_num == idx)[0]] for idx in range(1, len(bin_edges))])
            binned_sumweights2_num = np.asarray([np.sum(np.square(w)) for w in binned_weights_num])
            binned_sum2weights_num = np.asarray([np.square(np.sum(w)) for w in binned_weights_num])

            bin_indices_one_minus_num = np.digitize(one_minus_num_prediction, bin_edges)
            binned_weights_one_minus_num = np.asarray([one_minus_num_weights[np.where(bin_indices_one_minus_num == idx)[0]] for idx in range(1, len(bin_edges))])
            binned_sumweights2_one_minus_num = np.asarray([np.sum(np.square(w)) for w in binned_weights_one_minus_num])
            binned_sum2weights_one_minus_num = np.asarray([np.square(np.sum(w)) for w in binned_weights_one_minus_num])

            bin_indices_den = np.digitize(den_prediction, bin_edges)
            binned_weights_den = np.asarray([den_weights[np.where(bin_indices_den == idx)[0]] for idx in range(1, len(bin_edges))])
            binned_sum2weights_den = np.asarray([np.square(np.sum(w)) for w in binned_weights_den])

            efficiency_err = np.sqrt(binned_sumweights2_num * binned_sum2weights_one_minus_num + binned_sum2weights_num * binned_sumweights2_one_minus_num) / binned_sum2weights_den

            x_err = [(bin_edges[i+1]-bin_edges[i])/2 for i in range(len(bin_edges)-1)]
            ax.errorbar(bin_centers,
                             efficiency,
                             xerr=x_err,
                             yerr=efficiency_err,
                             fmt='-',
                             label=label,
                             )
        else:
            ax.plot(bin_centers,
                    efficiency,
                    label=label)

        # setting titles etc.
        if title is not None:
            title = (title + '\nMicroBooNE Preliminary')
        else:
            title = 'MicroBooNE Preliminary'
        ax.set_title(title, loc='left')
        ax.autoscale()

        if onebin:
            ax.set_xlabel("One bin")
            ax.set_ylabel("Entries")
            ax.tick_params(axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off
            ax.tick_params(axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off
        else:
            if xlabel is None:
                xlabel = variable
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Efficiency / {:.2f}".format(bin_width))
        plt.ylim(0, 1)
        plt.tight_layout()
