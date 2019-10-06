//! port from http://nlp.seas.harvard.edu/2018/04/03/attention.html
use std::time::Instant;

use tch;
use tch::{Reduction, kind, nn, nn::Module, nn::OptimizerConfig, Device, Tensor, IndexOp};
use tch::nn::{Optimizer, ModuleT};

pub struct Context<'a> {
    vs: &'a nn::VarStore,
    p: &'a nn::Path<'a>,
    device: Device,
    is_train: bool,
}

impl<'a> Context<'a> {
    fn new(vs: &'a nn::VarStore, p: &'a nn::Path, device: Device, is_train: bool) -> Self {
        Context {
            vs,
            p,
            device,
            is_train,
        }
    }
}

type EmbPosEnc<'a> = (Embeddings<'a>, PositionalEncoding<'a>);

// #[derive(Copy)]
pub struct EncoderDecoder<'a> {
    c: &'a Context<'a>,
    encoder: Encoder<'a>,
    decoder: Decoder<'a>,
    src_embed: EmbPosEnc<'a>,
    tgt_embed: EmbPosEnc<'a>,
    generator: Generator<'a>,
}

impl<'a> EncoderDecoder<'a>  {
    fn new(c: &'a Context, encoder: Encoder<'a>, decoder: Decoder<'a>, 
           src_embed: EmbPosEnc<'a>, tgt_embed: EmbPosEnc<'a>, 
           generator: Generator<'a>) -> Self {
        EncoderDecoder {
            c,
            encoder,
            decoder,
            src_embed,
            tgt_embed,
            generator,
        }
    }

    fn encode(&self, src: &Tensor, src_mask: &Tensor) -> Tensor {
        self.encoder.forward(&self.src_embed.1.forward(&self.src_embed.0.forward(src)), src_mask)
    }

    fn decode(&self, memory: &Tensor, src_mask: &Tensor, tgt: &Tensor, tgt_mask: &Tensor) -> Tensor {
        self.decoder.forward(&self.tgt_embed.1.forward(&self.tgt_embed.0.forward(tgt)), memory, src_mask, tgt_mask)
    }

    fn forward(&self, src: &Tensor, tgt: &Tensor, src_mask: &Tensor, tgt_mask: &Tensor) -> Tensor {
        self.decode(&self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    }
}

struct Generator<'a> {
    c: &'a Context<'a>,
    proj: nn::Linear,
}

impl<'a> Generator<'a> {
    fn new(c: &'a Context, p: nn::Path, d_model: usize, vocab: i64) -> Self {
        Generator {
            c,
            proj: nn::linear(p, d_model as i64, vocab, Default::default())
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        self.proj.forward(x).log_softmax(-1, kind::Kind::Float)
    }
}

struct Encoder<'a> {
    c: &'a Context<'a>,
    layers: Vec<EncoderLayer<'a>>,
    norm: LayerNorm<'a>,
}

impl<'a> Encoder<'a> {
    fn new(c: &'a Context, layers: Vec<EncoderLayer<'a>>, N: usize) -> Self {
        let size = &layers[0].size.clone();
        Encoder {
            c,
            layers,
            norm: LayerNorm::new(c, size, 1e-6),
        }
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Tensor {
        let mut x = x;
        let mut tmp;
        for layer in &self.layers {
            tmp = layer.forward(x, mask);
            x = &tmp;
        }
        self.norm.forward(&x)
    }
}

struct LayerNorm<'a> {
    c: &'a Context<'a>,
    a_2: Tensor,
    b_2: Tensor,
    eps: f64,
}

impl<'a> LayerNorm<'a> {
    fn new(c: &'a Context, features: &[i64], eps: f64) -> Self {
        LayerNorm {
            c,
            a_2: c.p.ones("a_2", features),
            b_2: c.p.zeros("b_2", features),
            eps,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mean = x.mean1(&[-1], true, kind::Kind::Float);
        let std = x.std1(&[-1], true, true);
        &self.a_2 * (x - mean) / (std + self.eps) + &self.b_2
    }
}

struct SublayerConnection<'a> {
    c: &'a Context<'a>,
    norm: LayerNorm<'a>,
    dropout: f64,
}

impl<'a> SublayerConnection<'a> {
    fn new(c: &'a Context, size: &[i64], dropout: f64) -> Self {
        SublayerConnection {
            c,
            norm: LayerNorm::new(c, size, 1e-6),
            dropout,
        }
    }

    fn forward(&self, x: &Tensor, sublayer: &'a dyn Fn(Tensor) -> Tensor) -> Tensor {
        x + sublayer(self.norm.forward(x)).dropout(self.dropout, self.c.is_train)
    }
}

struct EncoderLayer<'a> {
    c: &'a Context<'a>,
    self_attn: MultiHeadedAttention<'a>,
    feed_forward: PositionwiseFeedForward<'a>,
    sublayer: Vec<SublayerConnection<'a>>,
    size: Vec<i64>,
}

impl<'a> EncoderLayer<'a> {
    fn new(c: &'a Context, size: Vec<i64>, self_attn: MultiHeadedAttention<'a>, 
           feed_forward: PositionwiseFeedForward<'a>, dropout: f64) -> Self {
        EncoderLayer {
            c,
            self_attn,
            feed_forward,
            sublayer: (0..2).map(|_| SublayerConnection::new(c, &size.clone(), dropout)).collect(),
            size,
        }
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Tensor {
        let x = self.sublayer[0].forward(x, &|x| self.self_attn.forward(&x, &x, &x, Some(mask)));
        self.sublayer[1].forward(&x, &|x| self.feed_forward.forward(&x))
    }
}

struct Decoder<'a> {
    c: &'a Context<'a>,
    layers: Vec<DecoderLayer<'a>>,
    norm: LayerNorm<'a>,
}

impl<'a> Decoder<'a> {
    fn new(c: &'a Context, layers: Vec<DecoderLayer<'a>>, N: usize) -> Self {
        let size = &layers[0].size.clone();
        Decoder {
            c,
            layers,
            norm: LayerNorm::new(c, size, 1e-6),
        }
    }

    fn forward(&self, x: &Tensor, memory: &Tensor, src_mask: &Tensor, tgt_mask: &Tensor) -> Tensor {
        let mut x = x;
        let mut tmp;
        for layer in &self.layers {
            tmp = layer.forward(x, memory, src_mask, tgt_mask);
            x = &tmp;
        }
        self.norm.forward(x)
    }
}

struct DecoderLayer<'a> {
    c: &'a Context<'a>,
    size: Vec<i64>,
    self_attn: MultiHeadedAttention<'a>, 
    src_attn: MultiHeadedAttention<'a>, 
    feed_forward: PositionwiseFeedForward<'a>,
    sublayer: Vec<SublayerConnection<'a>>,
}

impl<'a> DecoderLayer<'a> {
    fn new(c: &'a Context, size: Vec<i64>, self_attn: MultiHeadedAttention<'a>, 
           src_attn: MultiHeadedAttention<'a>, feed_forward: PositionwiseFeedForward<'a>, dropout: f64) -> Self {
        let tmp = size.clone();
        DecoderLayer {
            c,
            size,
            self_attn,
            src_attn,
            feed_forward,
            sublayer: (0..3).map(|_| SublayerConnection::new(c, &tmp, dropout)).collect(),
        }
    }

    fn forward(&self, x: &Tensor, memory: &Tensor, src_mask: &Tensor, tgt_mask: &Tensor) -> Tensor {
        let m = memory;
        let x = &self.sublayer[0].forward(x, &|x| self.self_attn.forward(&x, &x, &x, Some(tgt_mask)));
        let x = self.sublayer[1].forward(x, &|x| self.src_attn.forward(&x, m, m, Some(src_mask)));
        self.sublayer[2].forward(&x, &|x| self.feed_forward.forward(&x))
    }
}

fn subsequent_mask<'a>(c: &'a Context, size: usize) -> Tensor {
    let attn_shape = &[1, size as i64, size as i64];
    let mask = Tensor::ones(attn_shape, (kind::Kind::Int64, c.device)).triu(1).totype(kind::Kind::Uint8);
    mask.eq(0)
}

fn attention<'a>(c: &'a Context, query: &Tensor, key: &Tensor, value: &Tensor, 
             mask: Option<&Tensor>, dropout: Option<f64>) -> (Tensor, Tensor) {
    let tmp = query.size();
    let d_k = tmp.last().unwrap();
    let mut scores = query.matmul(&(key.transpose(-2, -1) / (*d_k as f64).sqrt()));

    if let Some(m) = mask {
        scores = scores.masked_fill(&m.eq(0), -1e9);
    }
    let mut p_attn = scores.softmax(-1, kind::Kind::Float);
    if let Some(d) = dropout {
        p_attn = p_attn.dropout(d, c.is_train);
    }

    (p_attn.matmul(&value), p_attn)
}

struct MultiHeadedAttention<'a> {
    c: &'a Context<'a>,
    d_k: usize,
    h: usize,
    linears: Vec<nn::Linear>,
    attn: Option<Tensor>,
    dropout: Option<f64>,
}

impl<'a> MultiHeadedAttention<'a> {
    fn new(c: &'a Context, p: nn::Path, h: usize, d_model: usize, dropout: Option<f64>) -> Self {
        MultiHeadedAttention {
            c,
            d_k: d_model / h,
            h,
            linears: (0..4).map(|_| nn::linear(&p, d_model as i64, d_model as i64, Default::default())).collect(),
            attn: None,
            dropout
        }
    }

    fn forward(&self, query: &Tensor, key: &Tensor, 
               value: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let mut mask = mask; 
        let v; 
        if mask.is_some() {
            v = mask.unwrap().unsqueeze(1);
            mask = Some(&v);
        }
        let nbatches = query.size()[0];

        let xs: Vec<Tensor> = self.linears
                                .iter()
                                .zip(vec![query, key, value])
                                .map(|lx| {
                                    lx.0.forward(&lx.1)
                                        .view([nbatches, -1, self.h as i64, self.d_k as i64])
                                        .transpose(1, 2)
                                })
                                .collect();
        let (query, key, value) = (&xs[0], &xs[1], &xs[2]);
        let (x, attn) = attention(self.c, query, key, value, mask, self.dropout);
        // self.attn = Some(attn);
        let x = x.transpose(1, 2)
             .contiguous()
             .view([nbatches, -1, (self.h * self.d_k) as i64]);
        self.linears.last().unwrap().forward(&x)
    }
}

struct PositionwiseFeedForward<'a> {
    c: &'a Context<'a>,
    w_1: nn::Linear,
    w_2: nn::Linear,
    dropout: f64,
}

impl<'a> PositionwiseFeedForward<'a> {
    fn new(c: &'a Context, p: nn::Path, d_model: usize, d_ff: usize, dropout: f64) -> Self {
        PositionwiseFeedForward {
            c,
            w_1: nn::linear(&p, d_model as i64, d_ff as i64, Default::default()),
            w_2: nn::linear(&p, d_ff as i64, d_model as i64, Default::default()),
            dropout,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        self.w_2.forward(&self.w_1.forward(x).relu().dropout(self.dropout, self.c.is_train))
    }
}

struct Embeddings<'a> {
    c: &'a Context<'a>,
    lut: nn::Embedding,
    d_model: usize,
}

impl<'a> Embeddings<'a> {
    fn new(c: &'a Context, p: nn::Path, d_model: usize, vocab: i64) -> Self {
        Embeddings {
            c,
            lut: nn::embedding(p, vocab, d_model as i64, Default::default()),
            d_model,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        self.lut.forward(x) * (self.d_model as f64).sqrt()
    }
}

struct PositionalEncoding<'a> {
    c: &'a Context<'a>,
    dropout: f64,
    pe: Tensor,
}

impl<'a> PositionalEncoding<'a> {
    fn new(c: &'a Context, d_model: usize, dropout: f64, max_len: usize) -> Self {
        let mut pe = Tensor::zeros(&[max_len as i64, d_model as i64], kind::FLOAT_CPU);
        let position = Tensor::arange1(0, max_len as i64, kind::FLOAT_CPU).unsqueeze(1);
        let div_term = (Tensor::arange2(0, d_model as i64, 2, kind::FLOAT_CPU)
                        * -((10000.0 as f64).log(10.0) / d_model as f64)).exp();
        let idxes = Tensor::of_slice(&(0..d_model as i64).step_by(2).collect::<Vec<_>>());
        let idxes2 = Tensor::of_slice(&(1..d_model as i64).step_by(2).collect::<Vec<_>>());
        pe.i((.., &idxes)).copy_(&(&position * &div_term).sin());
        pe.i((.., &idxes2)).copy_(&(&position * &div_term).cos());
        let _ = pe.unsqueeze_(0);
        // self.register_buffer('pe', pe)
        PositionalEncoding {
            c,
            dropout,
            pe,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = x + self.pe.i((.., ..x.size()[1])).set_requires_grad(false);
        x.dropout(self.dropout, self.c.is_train)
    }
}

fn make_model<'a>(c: &'a Context, src_vocab: i64, tgt_vocab: i64, N: usize,
                  d_model: usize, d_ff: usize, h: usize, dropout: f64) -> EncoderDecoder<'a> {
    let src_position = PositionalEncoding::new(c, d_model, dropout, 5000);
    let tgt_position = PositionalEncoding::new(c, d_model, dropout, 5000);
    let model = EncoderDecoder::new(
        c,
        Encoder::new(c, create_encoder_layers(c, d_model, d_ff, h, dropout, N), N),
        Decoder::new(c, create_decoder_layers(c, d_model, d_ff, h, dropout, N), N),
        (Embeddings::new(c, c.p / "src_emb", d_model, src_vocab), src_position),
        (Embeddings::new(c, c.p / "tgt_emb", d_model, tgt_vocab), tgt_position),
        Generator::new(c, c.p / "gen", d_model, tgt_vocab),
    );

    tch::no_grad(|| {
        for mut p in c.vs.trainable_variables() {
            if p.dim() > 1 {
                p.init(nn::Init::KaimingUniform);
            }
        }
    });
    model
}

fn create_encoder_layers<'a>(c: &'a Context, d_model: usize, d_ff: usize, 
                             h: usize, dropout: f64, N: usize) -> Vec<EncoderLayer<'a>> {
    (0..N).map(move |i| {
        let v = vec![d_model as i64];
        let attn_enc = MultiHeadedAttention::new(c, c.p / format!("enc_att_{}", i), h, d_model, Some(dropout));
        let ff_enc = PositionwiseFeedForward::new(c, c.p / format!("enc_ff_{}", i), d_model, d_ff, dropout);
        EncoderLayer::new(c, v, attn_enc, ff_enc, dropout)
    }).collect()
}

fn create_decoder_layers<'a>(c: &'a Context, d_model: usize, d_ff: usize, 
                             h: usize, dropout: f64, N: usize) -> Vec<DecoderLayer<'a>> {
    (0..N).map(move |i| {
        let v = vec![d_model as i64];
        let self_attn_dec = MultiHeadedAttention::new(c, c.p / format!("dec_self_att_{}", i), h, d_model, Some(dropout));
        let src_attn_dec = MultiHeadedAttention::new(c, c.p / format!("dec_src_att_{}", i), h, d_model, Some(dropout));
        let ff_dec = PositionwiseFeedForward::new(c, c.p / format!("dec_ff_{}", i), d_model, d_ff, dropout);
        DecoderLayer::new(c, v, self_attn_dec, src_attn_dec, ff_dec, dropout)
    }).collect()
}

struct Batch<'a> {
    c: &'a Context<'a>,
    src: Tensor,
    src_mask: Tensor,
    trg: Option<Tensor>,
    trg_y: Option<Tensor>,
    trg_mask: Option<Tensor>,
    ntokens: Option<usize>,
}

impl<'a> Batch<'a> {
    fn new(c: &'a Context, src: Tensor, trg: Option<Tensor>, pad: usize) -> Self {
        let tmp = src.copy();
        let mut ret = Batch {
            c,
            src,
            src_mask: tmp.ne(pad as i64).unsqueeze(-2),
            trg: None,
            trg_y: None,
            trg_mask: None,
            ntokens: None,
        };
        if trg.is_some() {
            ret.trg = Some(trg.as_ref().unwrap().i((.., ..*trg.as_ref().unwrap().size().last().unwrap()-1)));
            ret.trg_y = Some(trg.as_ref().unwrap().i((.., 1..)));
            ret.trg_mask = Some(make_std_mask(c, ret.trg.as_ref().unwrap(), pad));
            // ret.ntokens = ret.trg_y.ne(pad as i64).data.sum();
            ret.ntokens = Some(ret.trg_y.as_ref().unwrap().ne(pad as i64).sum(kind::Kind::Int64).int64_value(&[]) as usize);
        }
        ret
    }
}

fn make_std_mask<'a>(c: &'a Context, tgt: &Tensor, pad: usize) -> Tensor {
    let tgt_mask = tgt.ne(pad as i64).unsqueeze(-2);
    // let tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data);
    let tgt_mask = tensor_and(&tgt_mask, &subsequent_mask(c, *tgt.size().last().unwrap() as usize).type_as(&tgt_mask));
    tgt_mask
}

fn tensor_and(a: &Tensor, b: &Tensor) -> Tensor {
    let tmp = Tensor::broadcast_tensors(&[a, b]);
    let (a, b) = (&tmp[0], &tmp[1]);
    let size = a.size();
    let ret = a.copy();

    for i in 0..size[0] {
        for j in 0..size[1] {
            for k in 0..size[2] {
                let tmp = i64::from(a.i((i, j, k))) & i64::from(b.i((i, j, k)));
                let _ = ret.i((i, j, k)).fill_(tmp);
            }
        }
    }
    ret
}

fn run_epoch<T>(data_iter: Vec<Batch>, model: &EncoderDecoder, 
                loss_compute: &mut SimpleLossCompute<T>) -> f64 {
    let mut start = Instant::now();
    let mut total_tokens = 0;
    let mut total_loss = 0.;
    let mut tokens = 0;

    let mut i = 0;
    for batch in data_iter.iter() {
        let out = model.forward(&batch.src, &batch.trg.as_ref().unwrap(), &batch.src_mask, &batch.trg_mask.as_ref().unwrap());
        let loss = loss_compute.call(&out, &batch.trg_y.as_ref().unwrap(), batch.ntokens.unwrap(), i); 
        total_loss += loss.double_value(&[]);
        total_tokens += batch.ntokens.unwrap();
        tokens += batch.ntokens.unwrap();

        if i % 50 == 1 {
            let elapsed = start.elapsed();
            println!("Epoch Step: {} Loss: {} Tokens per Sec: {}", 
                i, f64::from(&(loss / batch.ntokens.unwrap() as f64)), tokens as u64 / elapsed.as_secs());
            start = Instant::now();
            tokens = 0;
        }
        i += 1;
    }

    return total_loss / total_tokens as f64;
}

/// Optim wrapper that implements rate.
struct NoamOpt<T> {
    optimizer: Optimizer<T>,
    warmup: f64,
    factor: f64,
    model_size: usize,
}

impl<T> NoamOpt<T> {
    fn new(model_size: usize, factor: f64, warmup: f64, optimizer: Optimizer<T>) -> Self {
        NoamOpt {
            optimizer,
            warmup,
            factor,
            model_size,
        }
    }

    fn step(&mut self, step: u64) {
        let rate = self.rate(step);

        // for mut p in self.optimizer.trainable_variables.param_groups() {
        //     p.update("lr", rate);
        // }
        self.optimizer.set_lr(rate);
        self.optimizer.step();
    }

    fn rate(&self, step: u64) -> f64 {
        self.factor * (self.model_size as f64).powf(-0.5)
            * ((step as f64).powf(-0.5)).min(step as f64 * self.warmup.powf(-1.5))
    }
}

/*
fn get_std_opt<T>(model: &EncoderDecoder) -> NoamOpt<T> {
    NoamOpt::new(model.src_embed.0.d_model, 2 as f64, 4000 as f64, 
                 optimizer.adam(0.9, 0.98,  1e-9).build(0))
}
*/

struct LabelSmoothing {
    padding_idx: usize,
    confidence: f64,
    smoothing: f64,
    size: i64,
    ture_dist: Option<Tensor>,
}

impl LabelSmoothing {
    fn new(size: i64, padding_idx: usize, smoothing: f64) -> Self {
        LabelSmoothing {
            padding_idx,
            confidence: 1.0 - smoothing,
            smoothing,
            size,
            ture_dist: None,
        }
    }

    fn forward(&self, x: &Tensor, target: Tensor) -> Tensor {
        assert!(x.size()[1] == self.size);
        let mut ture_dist = x.copy();
        let _ = ture_dist.fill_(self.smoothing / (self.size - 2) as f64);
        let _ = ture_dist.scatter_1(1, &target.unsqueeze(1), self.confidence);
        let _ = ture_dist.i((.., self.padding_idx as i64)).fill_(0);
        // target.eq(self.padding_idx as i64).print();
        // ture_dist.index_fill_1(1, self.padding_idx, 0);
        let mask = target.eq(self.padding_idx as i64).nonzero();
        // println!("{:?}, {:?}", mask.size(), mask.squeeze1(0).size());
        if mask.dim() > 0 && mask.size()[0] > 0 {
            let _ = ture_dist.index_fill_(0, &mask.squeeze1(0), 0.0);
        }
        // self.ture_dist = Some(ture_dist.copy());
        x.kl_div(&ture_dist, Reduction::Sum)
    }
}

fn data_gen<'a>(c: &'a Context, V: i64, batch: i64, nbatches: usize) -> Vec<Batch<'a>> {
    (0..nbatches).map(|i| {
        let data = Tensor::randint1(1 as i64, V, &[batch, 10], (kind::Kind::Int64, c.device));
        let _ = data.i((.., 0)).fill_(1);
        Batch::new(c, data.copy(), Some(data.copy()), 0)
    }).collect()
}

struct SimpleLossCompute<'a, T> {
    generator: &'a Generator<'a>,
    criterion: &'a LabelSmoothing,
    // opt: Option<Optimizer<T>>,
    opt: Option<&'a mut NoamOpt<T>>,
}

impl<'a, T> SimpleLossCompute<'a, T> {
    fn new(generator: &'a Generator<'a>, criterion: &'a LabelSmoothing, opt: Option<&'a mut NoamOpt<T>>) -> Self {
        SimpleLossCompute {
            generator,
            criterion,
            opt,
        }
    }

    fn call(&mut self, x: &Tensor, y: &Tensor, norm: usize, step: u64) -> Tensor {
        let x = self.generator.forward(x);
        let loss = self.criterion.forward(&x.contiguous().view([-1, *x.size().last().unwrap()]), 
                                          y.contiguous().view(-1)) / norm as i64;
        if self.opt.is_some() {
            // self.opt.unwrap().optimizer.backward_step(&loss);
            self.opt.as_ref().unwrap().optimizer.zero_grad();
            loss.backward();
            self.opt.as_mut().unwrap().step(step);
        }

        // return loss.data[0] * norm;
        return loss * norm as i64;
    }
}

pub fn train() {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let p = vs.root();
    let c = Context::new(&vs, &p, device, true);
    _train(&c);
}

fn _train<'a>(c: &'a Context) -> EncoderDecoder<'a> {
    // let device = Device::cuda_if_available();
    // let vs = nn::VarStore::new(device);

    let V: i64 = 11;
    let model = make_model(&c, V, V, 2, 512, 2048, 8, 0.1);
    // let m = nn::func_t(|xs, train| {
        // let p = vs.root();
        // let c = Context::new(&vs, &p, device, train);
        let criterion = LabelSmoothing::new(V, 0, 0.0);
        let mut model_opt = NoamOpt::new(model.src_embed.0.d_model, 1 as f64, 400 as f64,
                                     nn::Adam::default().build(c.vs, 5e-1).unwrap());

        for epoch in 0..10 {
            let mut l = SimpleLossCompute::new(&model.generator, &criterion, Some(&mut model_opt));
            run_epoch(data_gen(&c, V, 30, 20), &model, &mut l);
            // run_epoch(data_gen(&c, V, 30, 5), &model, 
            //           SimpleLossCompute::new(&model.generator, &criterion, None as Option<&NoamOpt<nn::Adam>>));
        }
    // });

    // m.forward_t(&Tensor::ones(&[1, 1], (kind::Kind::Int64, c.device)), true);
    model
}

pub fn greedy_decode<'a>(c: &'a Context, model: EncoderDecoder, src: &Tensor, src_mask: &Tensor,
                         max_len: usize, start_symbol: i64) -> Tensor {
    let memory = model.encode(src, src_mask);
    let mut ys = Tensor::ones(&[1, 1], (kind::Kind::Int64, c.device))
                     .fill_(start_symbol)
                     .type_as(src);
    for i in 0..(max_len-1) {
        let out = model.decode(&memory, src_mask, &ys, 
                    &subsequent_mask(c, ys.size()[1] as usize).type_as(src));
        let prob = model.generator.forward(&out.i((.., -1)));
        let (_, mut next_word) = prob.max2(1, false);
        next_word = next_word.i(0);
        ys = Tensor::cat(&[ys, Tensor::ones(&[1, 1], (kind::Kind::Int64, c.device)).type_as(src).fill_1(&next_word)], 1);
    }
    ys
}

pub fn predict() {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let p = vs.root();
    let c = Context::new(&vs, &p, device, true);
    let model = _train(&c);

    let c = Context::new(&vs, &p, device, false);
    let src = Tensor::of_slice(&[1,2,3,4,5,6,7,8,9,10]).unsqueeze(0).totype(kind::Kind::Int64);
    let src_mask = Tensor::ones(&[1, 1, 10], (kind::Kind::Int64, c.device));
    greedy_decode(&c, model, &src, &src_mask, 10, 1).print();
}

pub fn main() {
    predict();
}
