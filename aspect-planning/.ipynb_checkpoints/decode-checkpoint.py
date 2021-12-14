
class BeamSearchNode(object):
    def __init__(self, hidden, prev_node, aspect_id, lop_p, length):
        self.hidden = hidden
        self.prevNode = prev_node
        self.aspect_id = aspect_id
        self.log_p = log_p
        self.len = length
        
        self.coherence = 0  # inject the term here
        self.generality = 0
        self.simplicity = 0

    def eval(self, alpha=1.0):
        alpha = 0 # overall influence
        w = {
            'coherence': 0.2,
            'generality': 0.4,
            'simplicity': 0.4
        }

        return self.log_p / float(self.len - 1 + 1e-6) \
            + alpha * (w['coherence'] * self.coherence + \
                       w['generality'] * self.generality + \
                       w['simplicity'] * self.simplicity)


decoder = DecoderRNN()


def beam_decode_for_condition(target_tensor, decoder_hiddens, encoder_outputs=None):
    '''
    :param target_tensor - batch information [B, T] (T: maximum output aspect length)
    :param decoder_hidden - input [1, B, H] as a starting point
    :param encoder_outputs
    :return decoded_batch
    '''

    beam_width = 10
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(target_tensor.size(0)):
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[SOS_token]], device=device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.aspect_id
            decoder_hidden = n.hidden

            if n.aspect_id.item() == EOS_token and n.prev_node != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            next_nodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.log_p + log_p, n.len + 1)
                score = -node.eval()
                next_nodes.append((score, node))

            # put them into queue
            for i in range(len(next_nodes)):
                score, nn = next_nodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(next_nodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.aspect_id)
            # back trace
            while n.prev_node != None:
                n = n.prev_node
                utterance.append(n.aspect_id)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch