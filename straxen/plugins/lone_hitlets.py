import strax
import numpy as np
import numba


export, __all__ = strax.exporter()

to_pe_file_tpc_default   = [1 for i in range(494)]
channel_list_tpc_default = [i for i in range(494)]

@export
@strax.takes_config(  
    strax.Option('to_pe_file_tpc', 
                 default=to_pe_file_tpc_default,
                 help='Expect gains in units ADC/sample.'),
    strax.Option('channel_list_tpc',
                 default=(tuple(channel_list_tpc_default)),
                 help="List of PMTs. Defalt value: all the PMTs")
)

class LoneHitlets(strax.Plugin):
    """
    Lone_hitlets.
    """
    depends_on = ('records', 'lone_hits')
    provides   = 'lone_hitlets'
    data_kind  = 'lone_hitlets'
    parallel   = 'process'
    compressor = 'lz4'

    __version__ = '0.0.1'
           
    def infer_dtype(self):
        self.dtype = strax.hitlet_dtype()
        return self.dtype
    
    def setup(self):
        self.to_pe        = np.array(self.config['to_pe_file_tpc'])
        self.channel_list = np.array(self.config['channel_list_tpc'])
    
    def compute(self, records, lone_hits, start, end):
        
        # Small test which ensures that lone_hits chunk had been split
        # according to orginal records chunks:
        res = wrong_confinment(records['time'], 
                               strax.endtime(records), 
                               lone_hits['time'], 
                               lone_hits['record_i'])
        if len(res) != np.sum(res):
            mes = f'Lh and records do not match {len(res)} and {np.sum(res)}.'
            raise ValueError(mes)
                        
        # Now convert hits into temp_hitlets including the data field:
        if len(lone_hits):
            nsamples = lone_hits['length'].max()
        else:
            nsamples = 0
        temp_hitlets = np.zeros(len(lone_hits), strax.hitlet_with_data_dtype(n_samples=nsamples))

        # Generating hitlets and copying relevant information from hits to hitlets.
        # These hitlets are not stored in the end since this array also contains a data
        # field which we will drop later.
        strax.refresh_hit_to_hitlets(lone_hits, temp_hitlets)
        del lone_hits

        # Get hitlet data and split hitlets:
        strax.get_hitlets_data(temp_hitlets, records, to_pe=self.to_pe)

        # Compute other hitlet properties:
        try:
            # Not sure yet why this fails for some lone_hits...
            strax.hitlet_properties(temp_hitlets)
        except:
            pass
        
        entropy = strax.conditional_entropy(temp_hitlets, template='flat', square_data=False)
        temp_hitlets['entropy'][:] = entropy
        strax.compute_widths(temp_hitlets)

        # Remove data field:
        lone_hitlets  = np.zeros(len(temp_hitlets), dtype=strax.hitlet_dtype())
        drop_data_field(temp_hitlets, lone_hitlets)
        
        return lone_hitlets
    
@numba.njit
def drop_data_field(old_hitlets, new_hitlets):
    """
    Function which copies everything except for the data field.
    If anyone know a better and faster way please let me know....
    :param old_hitlets:
    :param new_hitlets:
    :return:
    """
    n_hitlets = len(old_hitlets)
    for i in range(n_hitlets):
        o = old_hitlets[i]
        n = new_hitlets[i]

        n['time'] = o['time']
        n['length'] = o['length']
        n['dt'] = o['dt']
        n['channel'] = o['channel']
        n['hit_length'] = o['hit_length']
        n['area'] = o['area']
        n['amplitude'] = o['amplitude']
        n['time_amplitude'] = o['time_amplitude']
        n['entropy'] = o['entropy']
        n['width'][:] = o['width'][:]
        n['area_decile_from_midpoint'][:] = o['area_decile_from_midpoint'][:]
        n['fwhm'] = o['fwhm']
        n['fwtm'] = o['fwtm']
        n['left'] = o['left']
        n['low_left'] = o['low_left']
        n['record_i'] = o['record_i']  
        
@numba.njit()
def wrong_confinment(start, end, hitstart, record_i):
    
    res = np.zeros(len(record_i))
    for ind, ri in  enumerate(record_i):
        if start[ri] <= hitstart[ind] <= end[ri]:
            res[ind] = 1
    return res