from texttable import Texttable
import TableFactory as tf
from inspect import ismodule
from bayessb.multichain import MCMCSet
import cPickle
import inspect
import scipy.cluster.hierarchy
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from StringIO import StringIO
from collections import OrderedDict

reporter_dict = {}

class Report(object):
    """.. todo:: document this class """

    def __init__(self, chain_filenames, reporters, names=None, burn=0):
        """Create the Report object and run all reporter functions.

        Parameters
        ----------
        chain_filenames : OrderedDict of lists of MCMC filenames.
            The keys in the dict are the names of the groups of chains, sorted
            by name.  These should ideally be descriptive abbreviations,
            indicating the type of model, number of steps run in each chain,
            etc.  The entries in the dict are lists of filenames of pickled
            MCMC objects, representing completed MCMC estimation runs for the
            given model/data.
        reporters : mixed list of reporter functions and/or modules
            The reporter functions should take an instance of
            bayessb.MCMCSet.multichain as an argument and return an
            instance of pysb.report.Result. For inclusion in the report results
            table. If a module is included in the list, any reporter functions
            included in the module (i.e., functions decorated with
            @pysb.report.reporter) will be identified and applied to the
            chains.
        names : optional, list of strings
            Names to be used as the column headers in the report results
            table. If not provided, the keys from the chains dict are used
            as the column names.
        burn : int
            Number of steps to discard from the beginning of the chain.
        """

        self.chain_filenames = OrderedDict()
        """dict of lists of MCMC filenames, sorted by name."""
        for key in sorted(chain_filenames):
            self.chain_filenames[key] = chain_filenames[key]

        self.module_names = []
        """List of module names that parallels that of the reporter names."""
        self.reporters = []
        """List of reporter functions to run on the chains."""
        self.names = None
        """List of the different types of models/fits reported on."""
        self.results = []
        """List of lists containing the results of reporter execution."""
        self.burn = burn
        """Number of steps to discard from the beginning of the chain."""

        # Unpack reporter modules, adding any reporter functions found
        for reporter in reporters:
            if ismodule(reporter):
                self.reporters += reporter_dict[reporter.__name__]
                if hasattr(reporter, 'reporter_group_name'):
                    module_name = reporter.reporter_group_name
                else:
                    module_name = reporter.__name__
                self.module_names += [module_name] * \
                                     len(reporter_dict[reporter.__name__])
            else:
                self.reporters.append(reporter)
                # FIXME FIXME Fix this to sniff out the module for the
                # reporter function that was passed in
                self.module_names.append("Not given")

        # Initialize reporter names and module names
        if names is None:
            self.names = [n.replace('_', ' ')
                          for n in self.chain_filenames.keys()]
        else:
            self.names = names

        # Run the reports
        reporter_names = [r.reporter_name for r in self.reporters]
        for chain_list_name, chain_list in self.chain_filenames.iteritems():
            self.get_results_for_chain_set(chain_list_name, chain_list)

        # Transpose the results list 
        self.results = zip(*self.results)

    def get_results_for_chain_set(self, chain_list_name, chain_list):
        """Takes a list of filenames for a group of chains, initializes
        an MCMCSet object, and calls all of the reporters on the MCMCSet.
        Deferred the loading of MCMCSet objects to this function because
        it means that only one set of chains needs to be included in memory
        at any one time.
        """
        print "Loading chains for %s..." % chain_list_name
        mcmc_set = MCMCSet(chain_list_name)

        # Load the chain files
        for chain in chain_list:
            chain.load()

        # Prune and pool the chains in the list
        mcmc_set.initialize_and_pool(chain_list, self.burn)

        print "Running reporters for %s..." % chain_list_name
        result = []
        for reporter in self.reporters:
            result.append(reporter(mcmc_set))
        self.results.append(result)

    def get_text_table(self, max_width=80):
        """Return the report results as a pretty-printed text table."""
        # TODO This will have to be written because structure of results
        # table has changed
        tt = Texttable(max_width=max_width)
        tt.header(self.header_names)

        text_results = [[r.value if hasattr(r, 'value') else r for r in r_list]
                         for r_list in self.results]

        tt.add_rows(text_results, header=False)
        return tt.draw()

    def write_pdf_table(self, filename):
        """Writes the results table to a PDF file.

        Parameters
        ----------
        filename : string
            The name of the output filename.
        """
        # TODO This will have to be written because structure of results
        # table has changed
        lines = []
        for row in self.results:
            lines.append(tf.TableRow(*map(tf.Cell, row)))

        rowmaker = tf.RowSpec(*map(tf.ColumnSpec, self.header_names))
        table = tf.PDFTable('Results', headers=rowmaker)
        f = open(filename, 'wb')
        f.write(table.render(lines))

    def write_html_table(self, filename):
        """Writes the results table to a HTML file.

        Parameters
        ----------
        filename : string
            The name of the output filename.
        """

        lines = []
        for i, row in enumerate(self.results):
            html_row = []
            html_row.append(self.reporters[i].name)
            for result in row:
                # Here we assume it's a pysb.report.Result object
                if result.link is None:
                    html_row.append(result.value)
                else:
                    html_row.append('<a href=\'%s\'>%s</a>' %
                                    (result.link, result.value))

            lines.append(tf.TableRow(*map(tf.Cell, html_row)))

        rowmaker = tf.RowSpec(*map(tf.ColumnSpec, self.header_names))
        table = tf.HTMLTable('Results', headers=rowmaker)
        f = open(filename, 'wb')
        f.write(table.render(lines))

    def write_html_table_with_links(self, filename):
        """A manual re-write of HTML table export to allow inclusion of
        hyperlinks (the TableFactory version escapes the markup)
        """
        # Add some formatting for the overall page
        lines = """<!DOCTYPE html>
                <html>
                <head>
                    <style type="text/css">
                        body { font-family: sans-serif; font-size: 10pt}
                        table { border-collapse: collapse; }
                        th { align: left; font-weight: bold;
                             vertical-align: top}
                        td, th { border: 1px solid #aaa; padding: 0.2em; }
                    </style>
                </head>
                <body>"""

        lines += "<table>"

        # Add two empty columns to headers to allow for reporter and module
        headers = ['', ''] + self.names
        header_string = "<tr><th>"
        header_string += '</th><th>'.join(headers)
        header_string += "</th></tr>"
        lines += header_string

        prev_module_name = None
        for i, row in enumerate(self.results):
            html_row = []
            html_row_string = '<tr>'
            cur_module_name = self.module_names[i]

            # Group the results for a reporter group into a rowspan
            if prev_module_name != cur_module_name:
                rowspan = 1
                while (i + rowspan) < len(self.module_names) and \
                      self.module_names[i+rowspan] == cur_module_name:
                    rowspan += 1
                html_row_string += '<th rowspan="%d">%s</th>' % \
                                   (rowspan, cur_module_name)
            prev_module_name = cur_module_name

            # Add the row header showing the name of the current reporter.
            # If the reporter has an "evidence" field associated with it,
            # create a link to a page describing the evidence
            if (hasattr(self.reporters[i], 'reporter_evidence') and
                            self.reporters[i].reporter_evidence is not None):
                evidence_filename = '%s_evidence.html' % \
                                    self.reporters[i].__name__
                evidence_str = """
                    <html>
                    <head><title>Evidence for %s</title>
                    <style type="text/css">
                        img { max-width : 400px;
                              max-height : 400px; }
                        body { font-family: sans-serif; font-size: 10pt}
                        h1 { font-weight : bold;
                             font-size : 14pt; }
                    </style>
                    </head>
                    <body>
                    <p><h1>Evidence that %s</h1>
                """ % (self.reporters[i].reporter_name,
                       self.reporters[i].reporter_name)
                evidence_str += self.reporters[i].reporter_evidence.get_html()
                evidence_str += "</body></html>"

                with open(evidence_filename, 'w') as f:
                    f.write(evidence_str)

                reporter_html = '<th><a href="%s">%s</a></th>' % \
                                (evidence_filename,
                                self.reporters[i].reporter_name)
            else:
                reporter_html = '<th>%s</th>' % self.reporters[i].reporter_name
            html_row_string += reporter_html

            # Add the HTML-ified result
            for result in row:
                html_row.append(result.get_html())

            html_row_string += '\n'.join(html_row)
            html_row_string += '</tr>\n\n'
            lines += html_row_string
        lines += "</table>"

        # Add closing tags
        lines += "</body></html>"

        f = open(filename, 'wb')
        f.write(lines)

    def write_tsv(self, filename):
        """Write the transposed results table as a tab-separated file.

        Transposition of the results table puts the models in rows and the
        attributes in the columns, which is more suitable for most
        machine-learning/analysis algorithms.
        """

        output = StringIO()

        # Add the header line
        output.write('model_name\t')
        output.write('\t'.join([r.func_name for r in self.reporters]))
        output.write('\n')

        # Transpose the results list
        results = zip(*self.results)

        for model_name, result_row in zip(self.names, results):
            output.write(model_name + '\t')
            output.write('\t'.join([r.get_text() for r in result_row]))
            output.write('\n')

        with open(filename, 'w') as f:
            f.write(output.getvalue())

    def cluster_by_maximum_likelihood(self):
        """Cluster the models based on maximum_likelihood."""

        # Get the maximum likelihood row from the results table
        ml_results = None
        for i, reporter in enumerate(self.reporters):
            if reporter.func_name == "maximum_likelihood":
                ml_results = self.results[i]
        if ml_results is None:
            raise Exception("Couldn't find the row in the "
                            "results table for the maximum likelihood "
                            "test.")

        # Get the maximum likelihood row from the results table
        tBidBax_monotonic_results = None
        for i, reporter in enumerate(self.reporters):
            if reporter.func_name == "tBid_Bax_monotonically_increasing":
                tBidBax_monotonic_results = self.results[i]
        if tBidBax_monotonic_results is None:
            raise Exception("Couldn't find the row in the "
                            "results table for the tBidBax monotonically "
                            "increasing test.")

        # Get the maximum likelihood row from the results table
        iBax_monotonic_results = None
        for i, reporter in enumerate(self.reporters):
            if reporter.func_name == "iBax_monotonically_increasing":
                iBax_monotonic_results = self.results[i]
        if iBax_monotonic_results is None:
            raise Exception("Couldn't find the row in the "
                            "results table for the iBax monotonically "
                            "increasing test.")

        total_ml = np.sum([r.value for r in ml_results])
        combined_results = []
        # FIXME Need a better way of normalizing the max likelihood
        #combined_results.append([r.value / total_ml for r in ml_results])
        combined_results.append([np.log10(r.value/2) for r in ml_results])
        combined_results.append([r.value for r in tBidBax_monotonic_results])
        combined_results.append([r.value for r in iBax_monotonic_results])
        combined_results = np.array(combined_results)

        # Calculate distances between models based on tests
        num_results = combined_results.shape[1]
        D1 = scipy.zeros([num_results, num_results])
        for i in range(num_results):
            for j in range(num_results):
                D1[i, j] = np.linalg.norm(combined_results[:,i] -
                                         combined_results[:,j])

        # Compute and plot first dendrogram
        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
        Y = scipy.cluster.hierarchy.linkage(D1, method='centroid')
        Z1 = scipy.cluster.hierarchy.dendrogram(Y, orientation='right',
                labels=[n.split(' ')[0] + ' ' + n.split(' ')[2]
                        for n in self.names])
        ax1.set_xticks([])
        ax1.yaxis.tick_left()

        # Calculate distances between tests based on models
        num_results = combined_results.shape[0]
        D2 = scipy.zeros([num_results, num_results])
        for i in range(num_results):
            for j in range(num_results):
                D2[i, j] = np.linalg.norm(combined_results[i,:] -
                                          combined_results[j,:])

        # Compute and plot second dendrogram
        ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
        Y = scipy.cluster.hierarchy.linkage(D2, method='centroid')
        Z2 = scipy.cluster.hierarchy.dendrogram(Y,
                labels=['Max likelihood', 'tBidBax monotonic',
                        'iBax monotonic'])
        ax2.xaxis.tick_top()
        ax2.set_yticks([])

        # Plot distance matrix.
        axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
        idx1 = Z1['leaves']
        idx2 = Z2['leaves']
        D = np.zeros(combined_results.shape)
        for i in range(combined_results.shape[0]):
            for j in range(combined_results.shape[1]):
                D[i,j] = combined_results[idx2[i], idx1[j]]
        im = axmatrix.matshow(D.T, aspect='auto', origin='lower',
                              cmap=cm.YlGnBu)
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])

        # Plot colorbar.
        axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
        plt.colorbar(im, cax=axcolor)
        fig.show()
        fig.savefig('dendrogram.png')

class Result(object):
    """Stores the results associated with the execution of a reporter function.

    Parameters
    ----------
    value : anything
        The return value of a reporter function.
    link : string
        String representing a hyperlink, e.g. to information or
        visualizations supporting the reporter result.
    expectation : anything (optional)
        The expected value of the reporter.
    """
    def __init__(self, value, link, expectation=None):
        self.value = value
        self.link = link
        self.expectation = expectation

    def get_text(self):
        """Returns a text representation of the result."""
        result_str = ''
        if isinstance(self.value, float):
            result_str = '%-.2f' % self.value
        elif isinstance(self.value, bool):
            if self.value:
                result_str = 'True'
            else:
                result_str = 'False'
        else:
            result_str = str(self.value)

        return result_str

    def get_html(self):
        """Returns the default HTML string for the table cell to contain the
        result.

        Returns
        -------
        string
            A string containing the HTML for the table cell, including the
            opening and closing (<td>...</td>) tags.
        """
        # Format the result
        result_str = self.get_text()
        if self.link is not None:
            result_str = '<a href="%s">%s</a>' % (self.link, result_str)
        return '<td>%s</td>' % result_str

class FloatListResult(Result):
    """Implements formatting for a list of floating point values.

    In particular, specifies the precision at which they should be displayed.

    Parameters
    ----------
    value : anything
        The return value of a reporter function.
    link : string
        String representing a hyperlink, e.g. to information or
        visualizations supporting the reporter result.
    precision : int (optional)
        The number of decimal places to display for each entry in the
        list of values. Default is 2.
    """

    def __init__(self, value, link, precision=2):
        Result.__init__(self, value, link)
        self.precision = precision

    def get_text(self):
        """Returns the text representation of the floating point list.

        The string representation of the floating point list is of the form
        "<td>[xxx.xx, x.xx, xx.xx, ...]</td>, where the precision (number of
        x's after decimal point) is controlled by the value assigned to the
        property ``precision``.
        """
        # FIXME Finish comments
        if self.value is None:
            result_str = str(self.value)
        else:
            format_str = '%%.%dg' % self.precision
            result_str = '['
            result_str += ', '.join([format_str % f for f in self.value])
            result_str += ']'
        return result_str

    def get_html(self):
        """Returns the HTML string for the table cell to contain the result.

        The string representation of the floating point list is of the form
        "<td>[xxx.xx, x.xx, xx.xx, ...]</td>, where the precision (number of
        x's after decimal point) is controlled by the value assigned to the
        property ``precision``.

        Returns
        -------
        string
            A string containing the HTML for the table cell, including the
            opening and closing (<td>...</td>) tags.
        """
        result_str = self.get_text()
        if self.link is not None:
            result_str = '<a href="%s">%s</a>' % (self.link, result_str)
        return '<td>%s</td>' % result_str

class ThumbnailResult(Result):
    """A result that is an img that should be displayed as a thumbnail.

    Results of this type have no value associated with them, so the ``value``
    field is set to ``None``.
    """
    def __init__(self, thumbnail_link, img_link):
        """Create the FloatListResult object.

        Parameters
        ----------
        thumbnail_link : string
            Path to the filename of the thumbnail image.
        img_link : string
            Path to the filename of the full-size image.
        """
        if thumbnail_link is None or img_link is None:
            raise ValueError("Arguments to ThumbnailResult.__init__() "
                             "cannot be None.")

        Result.__init__(self, None, img_link)
        self.thumbnail_link = thumbnail_link

    def get_html(self):
        """Returns the HTML string for the table cell to contain the result.

        The string representation for the thumbnail is of the form
        ``<td><a href="..."><img ...></a></td>``, with the anchor tag
        linking to the full-size image.

        Returns
        -------
        string
            A string containing the HTML for the table cell, including the
            opening and closing (<td>...</td>) tags.
        """
        return '<td><a href="%s"><img src="%s" /></a></td>' % \
               (self.link, self.thumbnail_link)

class MeanSdResult(Result):
    """A result whose value is expressed as a mean and standard deviation.

    For example, to summarize a distribution which can be viewed by accessing
    the associated link. For these results, the "value" attribute is set to
    the mean; the SD is stored in the additional attribute ``sd``.

    Parameters
    ----------
    mean : float
        The mean value associated with the result.
    sd : float
        The standard deviation associated with the result.
    link : string
        Path to the filename of any additional data.
    precision : int (optional)
        The number of decimal places to use when displaying the mean
        and standard deviation. Default is 3.
    """

    def __init__(self, mean, sd, link, precision=3):
        if mean is None or sd is None or sd < 0:
            raise ValueError("Invalid argument to MeanSdResult constructor.")

        Result.__init__(self, mean, link)
        self.sd = sd
        self.precision = precision

    def get_html(self):
        """Returns the HTML string for the table cell to contain the result.

        The string representation for the thumbnail is of the form
        ``<td><a href="...">mean &plusmn; sd</a></td>``, with the anchor tag
        linking to the full-size image.

        Returns
        -------
        string
            A string containing the HTML for the table cell, including the
            opening and closing (<td>...</td>) tags.
        """

        format_str = '<td><a href="%%s">%%.%dg &plusmn; %%.%dg</a></td>' % \
                     (self.precision, self.precision)
        return format_str % (self.link, self.value, self.sd)

class FuzzyBooleanResult(Result):
    """Stores the result of a yes/no test applied to a chain by sampling.

    Color-codes the result of the boolean test as red (bad) or green (good)
    depending on its deviation from the expected value.
    """

    def __init__(self, value, link, expectation):
        if value is None or expectation is None:
            raise ValueError("value and expectation arguments cannot be None.")
        if not isinstance(value, float):
            raise ValueError("value must be a float.")
        if not isinstance(expectation, float):
            raise ValueError("value must be a float.")
        Result.__init__(self, value, link, expectation)

    def get_html(self):
        """Returns the default HTML string for the table cell to contain the
        result.

        Returns
        -------
        string
            A string containing the HTML for the table cell, including the
            opening and closing (<td>...</td>) tags.
        """
        # Format the result
        result_str = '%-.2f' % self.value
        if self.link is not None:
            result_str = '<a href="%s">%s</a>' % (self.link, result_str)
        error = abs(self.value - self.expectation)
        c_map = cm.get_cmap('RdYlGn')
        rgba = c_map(1 - error)
        color_str = "#%02x%02x%02x" % tuple([v*255 for v in rgba[0:3]])
        return '<td style="background: %s">%s</td>' % (color_str, result_str)

# DECORATOR
def reporter(name, evidence=None):
    """Decorator for reporter functions.

    Sets the ``name`` field of the function to indicate its name. The name of
    the reporter function is meant to be a human-readable name for use in
    results summaries.

    The decorator also adds the reporter function to the package-level variable
    ``reporter_dict``, which keeps track of all reporter functions imported (and
    decorated) thus far. The ``reporter_dict`` is indexed by the name of the
    module containing the reporter function, and each key maps to a list of 
    reporter functions.

    Parameters
    ----------
    name : string
        The human-readable name for the reporter function.
    evidence : Evidence
        The evidence for the reporter.

    Returns
    -------
    The decorated reporter function.
    """

    if callable(name):
        raise TypeError("The reporter decorator requires a name argument.")
    def wrap(f):
        # Keep track of all reporters in the package level reporter_dict
        reporter_mod_name = inspect.getmodule(f).__name__
        reporter_list = reporter_dict.setdefault(reporter_mod_name, [])
        reporter_list.append(f)
        f.reporter_name = name
        f.reporter_evidence = evidence
        return f
    return wrap
