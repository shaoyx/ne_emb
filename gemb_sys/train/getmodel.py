from __future__ import print_function

import vctrainer, deepwalk, app, combine, rw2vc
import generalwalk

def _getmodel(model, g, args):
    if model == 'deepwalk':
        return deepwalk.deepwalk(graph=g, fac=args.epoch_fac, window=args.window_size,
                                 degree_bound=args.degree_bound, degree_power=args.degree_power)
    if model == 'app':
        return app.APP(graph=g, jump_factor=args.app_jump_factor, sample=args.epoch_fac, step=args.app_step)

    if model == 'deepwalk,app':
        return combine.combine(g, args)

    if model == 'generalwalk':
        return generalwalk.generalwalk(g, fac=args.epoch_fac, window=args.window_size,
                                       degree_bound=args.degree_bound, degree_power=args.degree_power)

    if model == 'rw2vc':
        return rw2vc.rw2vc(graph=g, rw_file=args.rw_file,
                           window=args.window_size, emb_model=args.emb_model, rep_size=args.representation_size,
                           epoch=args.epochs, batch_size=args.batch_size,
                           learning_rate=args.lr, negative_ratio=args.negative_ratio)

    model_list = ['app', 'deepwalk', 'deepwalk,app', 'rw2vc', 'generalwalk']
    print ("The sampling method {} does not exist!", model)
    print ("Please choose from the following:")
    for m in model_list:
        print(m)
    exit()

def getmodels(g, args):

    model_v = _getmodel(args.model_v, g, args)

    if args.model_v == 'rw2vc':
        return model_v

    if not args.model_c:
        model_c = model_v
    elif args.model_c == args.model_v:
        model_c = model_v
    else:
        model_c = _getmodel(args.model_c, g, args)

    if not args.emb_model:
        arg_emb_model = "asym"
    else:
        arg_emb_model = args.emb_model

    trainer = vctrainer.vctrainer(g, model_v, model_c, emb_model=arg_emb_model, rep_size=args.representation_size,
                                  epoch=args.epochs, batch_size=args.batch_size,
                                  learning_rate=args.lr, negative_ratio=args.negative_ratio)
    return trainer
