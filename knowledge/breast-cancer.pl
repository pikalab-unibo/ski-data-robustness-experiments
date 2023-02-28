diagnosis(BareNuclei, BlandChromatin, ClumpThickness, MarginalAdhesion, Mitoses, NormalNucleoli, SingleEpithelialCellSize, UniformityCellShape, UniformityCellSize, 0) :-
    UniformityCellSize < 3, NormalNucleoli > 2, BareNuclei < 1.
diagnosis(BareNuclei, BlandChromatin, ClumpThickness, MarginalAdhesion, Mitoses, NormalNucleoli, SingleEpithelialCellSize, UniformityCellShape, UniformityCellSize, 1) :-
    UniformityCellSize < 3, NormalNucleoli > 2, BareNuclei > 1.
diagnosis(BareNuclei, BlandChromatin, ClumpThickness, MarginalAdhesion, Mitoses, NormalNucleoli, SingleEpithelialCellSize, UniformityCellShape, UniformityCellSize, 0) :-
    UniformityCellSize > 3, BareNuclei < 0.
diagnosis(BareNuclei, BlandChromatin, ClumpThickness, MarginalAdhesion, Mitoses, NormalNucleoli, SingleEpithelialCellSize, UniformityCellShape, UniformityCellSize, 1) :-
    UniformityCellSize > 3, BareNuclei > 0, BlandChromatin > 4.
diagnosis(BareNuclei, BlandChromatin, ClumpThickness, MarginalAdhesion, Mitoses, NormalNucleoli, SingleEpithelialCellSize, UniformityCellShape, UniformityCellSize, 1) :-
    UniformityCellSize > 3, BareNuclei > 0, BlandChromatin < 4, UniformityCellSize < 4, NormalNucleoli < 1.
diagnosis(BareNuclei, BlandChromatin, ClumpThickness, MarginalAdhesion, Mitoses, NormalNucleoli, SingleEpithelialCellSize, UniformityCellShape, UniformityCellSize, 0) :-
    UniformityCellSize > 3, BareNuclei > 0, BlandChromatin < 4, UniformityCellSize < 4, NormalNucleoli > 1.
diagnosis(BareNuclei, BlandChromatin, ClumpThickness, MarginalAdhesion, Mitoses, NormalNucleoli, SingleEpithelialCellSize, UniformityCellShape, UniformityCellSize, 1) :-
    UniformityCellSize > 3, BareNuclei > 0, BlandChromatin < 4, UniformityCellSize > 4, MarginalAdhesion > 1.
diagnosis(BareNuclei, BlandChromatin, ClumpThickness, MarginalAdhesion, Mitoses, NormalNucleoli, SingleEpithelialCellSize, UniformityCellShape, UniformityCellSize, 1) :-
    UniformityCellSize > 3, BareNuclei > 0, BlandChromatin < 4, UniformityCellSize > 4, MarginalAdhesion < 1, NormalNucleoli < 3.
diagnosis(BareNuclei, BlandChromatin, ClumpThickness, MarginalAdhesion, Mitoses, NormalNucleoli, SingleEpithelialCellSize, UniformityCellShape, UniformityCellSize, 0) :-
    UniformityCellSize > 3, BareNuclei > 0, BlandChromatin < 4, UniformityCellSize > 4, MarginalAdhesion < 1, NormalNucleoli > 3.
diagnosis(BareNuclei, BlandChromatin, ClumpThickness, MarginalAdhesion, Mitoses, NormalNucleoli, SingleEpithelialCellSize, UniformityCellShape, UniformityCellSize, 0) :-
    NormalNucleoli < 2, BareNuclei < 4.
diagnosis(BareNuclei, BlandChromatin, ClumpThickness, MarginalAdhesion, Mitoses, NormalNucleoli, SingleEpithelialCellSize, UniformityCellShape, UniformityCellSize, 1) :-
    BareNuclei > 4, SingleEpithelialCellSize < 1.
diagnosis(BareNuclei, BlandChromatin, ClumpThickness, MarginalAdhesion, Mitoses, NormalNucleoli, SingleEpithelialCellSize, UniformityCellShape, UniformityCellSize, 0) :-
    SingleEpithelialCellSize > 1.