from collections import defaultdict
from re import T

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

from .vilmodel import BertLayerNorm, BertOnlyMLMHead
from .vilmodel import NavPreTrainedModel
from .gelunits import Reg_Loss, GIoU_Loss, xyxy2xywh

class NextActionPrediction(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class NextActionRegression(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 3))

    def forward(self, x):
        return self.net(x)

class SpatialRelRegression(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size*2, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 2))

    def forward(self, x):
        return self.net(x)

class RegionClassification(nn.Module):
    " for MRC(-kl)"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output

class ItmPrediction(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class BboxPrediction(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 4))

    def forward(self, x):
        return self.net(x)

class SpanPrediction(nn.Module):
    def __init__(self, hidden_size, class_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, class_size))

    def forward(self, x):
        return self.net(x)


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=False, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(~mask.bool(), -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


class MultiStepNavCMTPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.bert = NavPreTrainedModel(config)

        if 'mlm' in config.pretrain_tasks:
            self.mlm_head = BertOnlyMLMHead(self.config)
        if 'sap' in config.pretrain_tasks:
            self.next_action = NextActionPrediction(self.config.hidden_size, self.config.pred_head_dropout_prob)
        if 'sar' in config.pretrain_tasks:
            self.regress_action = NextActionRegression(self.config.hidden_size, self.config.pred_head_dropout_prob)
        if 'sprel' in config.pretrain_tasks:
            self.sprel_head = SpatialRelRegression(self.config.hidden_size, self.config.pred_head_dropout_prob)
        if 'mrc' in config.pretrain_tasks:
            self.image_classifier = RegionClassification(self.config.hidden_size, self.config.image_prob_size)
        if 'itm' in config.pretrain_tasks:
            self.itm_head = ItmPrediction(self.config.hidden_size)
        if 'gel' in config.pretrain_tasks:
            self.land_att = SoftDotAttention(self.config.hidden_size, self.config.hidden_size)
            self.span_att = SoftDotAttention(self.config.hidden_size, self.config.hidden_size)
            self.bbox_head = BboxPrediction(self.config.hidden_size, self.config.gel_pred_head_dropout_prob)
            self.span_head = SpanPrediction(self.config.hidden_size, self.config.max_txt_len, self.config.gel_pred_head_dropout_prob)
            self.con_projection_image = nn.Linear(self.config.hidden_size, self.config.con_hdim)
            self.con_projection_text = nn.Linear(self.config.hidden_size, self.config.con_hdim)
        
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        if 'mlm' in self.config.pretrain_tasks:
            self._tie_or_clone_weights(self.mlm_head.predictions.decoder,
                self.bert.embeddings.word_embeddings)

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task.startswith('mlm'):
            return self.forward_mlm(batch['txt_ids'], batch['txt_masks'], 
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['txt_labels'], compute_loss)
        elif task.startswith('sap'):
            return self.forward_sap(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['ob_img_fts'], batch['ob_ang_fts'], 
                                    batch['ob_nav_types'], batch['ob_masks'],
                                    batch['ob_action_viewindex'], compute_loss)
        elif task.startswith('sar'):
            return self.forward_sar(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['ob_img_fts'], batch['ob_ang_fts'], 
                                    batch['ob_nav_types'], batch['ob_masks'],
                                    batch['ob_action_angles'], batch['ob_progress'], compute_loss)
        elif task.startswith('sprel'):
            return self.forward_sprel(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['ob_img_fts'], batch['ob_ang_fts'], 
                                    batch['ob_nav_types'], batch['ob_masks'],
                                    batch['sp_anchor_idxs'], batch['sp_targets'], 
                                    compute_loss)
        elif task.startswith('mrc'):
            return self.forward_mrc(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['hist_mrc_masks'], batch['hist_img_probs'], compute_loss)
        elif task.startswith('itm'):
            return self.forward_itm(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'], 4, compute_loss)
        elif task.startswith('gel'):
            return self.forward_bbox(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['ob_img_fts'], batch['ob_ang_fts'], 
                                    batch['ob_nav_types'], batch['ob_masks'],
                                    batch['span_gt'], batch['landmark_bbox'], batch['landmark_gt'],
                                    compute_loss)
        else:
            raise ValueError('invalid task')

    def forward_mlm(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    txt_labels, compute_loss):
        txt_embeds, _, _ = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            None, None, None, None)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(txt_embeds, txt_labels != -1)
        prediction_scores = self.mlm_head(masked_output)

        if compute_loss:
            mask_loss = F.cross_entropy(prediction_scores, 
                                        txt_labels[txt_labels != -1], 
                                        reduction='none')
            return mask_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_sap(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks, 
                    act_labels, compute_loss):
        txt_embeds, hist_embeds, ob_embeds = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks)
        
        # combine text and visual to predict next action
        prediction_scores = self.next_action(ob_embeds * txt_embeds[:, :1]).squeeze(-1)
        prediction_scores.masked_fill_(ob_nav_types == 0, -float('inf'))

        if compute_loss:
            act_loss = F.cross_entropy(prediction_scores, act_labels, reduction='none')
            return act_loss
        else:
            return prediction_scores

    def forward_sar(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks, 
                    ob_act_angles, ob_progress, compute_loss):
        txt_embeds, hist_embeds, ob_embeds = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks)

        prediction_scores = self.regress_action(txt_embeds[:, 0])   # [CLS] token

        if compute_loss:
            act_targets = torch.cat([ob_act_angles, ob_progress.unsqueeze(1)], dim=1)
            act_loss = F.mse_loss(prediction_scores, act_targets, reduction='none')
            return act_loss
        else:
            return prediction_scores

    def forward_sprel(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks, 
                    sp_anchor_idxs, sp_targets, compute_loss):
        txt_embeds, hist_embeds, ob_embeds = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks)

        # img_embeds: (batch, views, dim), sp_anchor_idxs: (batch)
        anchor_ob_embeds = torch.gather(ob_embeds, 1, 
            sp_anchor_idxs.unsqueeze(1).unsqueeze(2).repeat(1, 36, ob_embeds.size(-1)))
        # (batch, 1, dim)
        cat_ob_embeds = torch.cat([anchor_ob_embeds, ob_embeds[:, :-1]], -1)
        
        prediction_scores = self.sprel_head(cat_ob_embeds) # (batch, 36, 2)

        if compute_loss:
            sprel_loss = F.mse_loss(prediction_scores, sp_targets, reduction='none')
            return sprel_loss
        else:
            return prediction_scores

    def forward_mrc(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    hist_mrc_masks, hist_img_probs, compute_loss=True):
        txt_embeds, hist_embeds, _ = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            None, None, None, None)

        # only compute masked regions for better efficient=cy
        hist_embeds = hist_embeds[:, 1:] # remove global embedding
        masked_output = self._compute_masked_hidden(hist_embeds, hist_mrc_masks)
        prediction_soft_labels = self.image_classifier(masked_output)

        hist_mrc_targets = self._compute_masked_hidden(hist_img_probs, hist_mrc_masks)

        if compute_loss:
            prediction_soft_labels = F.log_softmax(prediction_soft_labels, dim=-1)
            mrc_loss = F.kl_div(prediction_soft_labels, hist_mrc_targets, reduction='none').sum(dim=1)
            return mrc_loss
        else:
            return prediction_soft_labels, hist_mrc_targets

    def forward_itm(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    num_neg_trajs, compute_loss):
        # (batch_size, 1+num_negs, dim)
        fused_embeds = self.bert.forward_itm(
            txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            num_neg_trajs=num_neg_trajs)

        prediction_scores = self.itm_head(fused_embeds).squeeze(2) # (batch, 1+num_negs, 1)
        # The first is positive
        itm_targets = torch.zeros(fused_embeds.size(0), dtype=torch.long).to(self.device)

        if compute_loss:
            sprel_loss = F.cross_entropy(prediction_scores, itm_targets, reduction='none')
            return sprel_loss
        else:
            return prediction_scores, itm_targets
    
    def forward_bbox(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks, 
                    span_gt, landmark_bbox, landmark_gt, compute_loss):
        txt_embeds, hist_embeds, ob_embeds = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks)

        tot_loss = torch.zeros(txt_embeds.size(0), dtype=torch.float).to(self.device)

        if self.config.loss_ce:
            landmark_embeds = torch.matmul(landmark_gt.unsqueeze(1), ob_embeds).squeeze() / landmark_gt.sum(1).unsqueeze(1)
            if self.config.attention:
                landmark_embeds, att = self.land_att(landmark_embeds, txt_embeds[:,1:,:], txt_masks[:,1:])
            pred_span = self.span_head(landmark_embeds)
            logits = pred_span.log_softmax(-1)
            target_sim = torch.zeros_like(logits)
            target_sim[:, :span_gt.size(1)] = span_gt
            ce_loss = -(logits * target_sim).sum(-1)
            # ce_loss = ce_loss.mean()
            tot_loss += ce_loss * self.config.weight_ce
        else:
            ce_loss = torch.zeros(txt_embeds.size(0), dtype=torch.float).to(self.device)

        if self.config.loss_bbox:
            span_embeds = torch.matmul(span_gt.unsqueeze(1), txt_embeds).squeeze() / span_gt.sum(1).unsqueeze(1)
            if self.config.attention:
                span_embeds, att = self.span_att(span_embeds, ob_embeds, ob_masks)
            pred_bbox = self.bbox_head(span_embeds).sigmoid() * 2.    #xywh

            GIoU_loss = GIoU_Loss(pred_bbox, landmark_bbox, 2)
            tot_loss += GIoU_loss * self.config.weight_giou
            gt_bbox_ = xyxy2xywh(landmark_bbox)
            l1_loss = Reg_Loss(pred_bbox, gt_bbox_)
            tot_loss += l1_loss * self.config.weight_l1
        else:
            GIoU_loss = torch.zeros(txt_embeds.size(0), dtype=torch.float).to(self.device)
            l1_loss = torch.zeros(txt_embeds.size(0), dtype=torch.float).to(self.device)

        if self.config.loss_con:
            positive_map = torch.matmul(landmark_gt.unsqueeze(2), span_gt.unsqueeze(1)).bool()
            proj_ob = F.normalize(self.con_projection_image(ob_embeds), p=2, dim=-1)
            proj_txt = F.normalize(self.con_projection_text(txt_embeds), p=2, dim=-1)
            logits = (
                torch.matmul(proj_ob, proj_txt.transpose(-1, -2)) / self.config.temperature
            ) 

            positive_logits = -logits.masked_fill(~positive_map, 0)
            negative_logits = logits  # .masked_fill(positive_map, -1000000)

            boxes_with_pos = positive_map.any(2)
            pos_term = positive_logits.sum(2)
            neg_term = negative_logits.logsumexp(2)

            nb_pos = positive_map.sum(2) + 1e-6

            box_to_token_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~boxes_with_pos, 0).mean()

            tokens_with_pos = positive_map.any(1)
            pos_term = positive_logits.sum(1)
            neg_term = negative_logits.logsumexp(1)

            nb_pos = positive_map.sum(1) + 1e-6

            tokens_to_boxes_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~tokens_with_pos, 0).mean()
            con_loss = ((box_to_token_loss + tokens_to_boxes_loss) / 2)
            tot_loss += con_loss * self.config.weight_con
        else:
            con_loss = torch.zeros(txt_embeds.size(0), dtype=torch.float).to(self.device)    
        
        if compute_loss:
            return tot_loss
        else:
            return ce_loss, GIoU_loss, l1_loss, con_loss
