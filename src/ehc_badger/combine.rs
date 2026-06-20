//! Combine step (EHC step 3) over [`Block`] using paper matrices.

use crate::block::Block;

/// Reference `Combine2` (`HalftimeHash16`).
#[inline(always)]
pub(crate) fn combine2<B: Block>(input: &[B; 7]) -> [B; 2] {
    let mut output = [input[0], input[1]];
    output[0] = output[0].plus(input[2]);
    output[1] = output[1].plus(input[2]);
    output[0] = output[0].plus(input[3]);
    output[1] = output[1].plus(input[3].shl::<1>());
    output[0] = output[0].plus(input[4].shl::<1>());
    output[1] = output[1].plus(input[4]);
    output[0] = output[0].plus(input[5]);
    output[1] = output[1].plus(input[5].shl::<2>());
    output[0] = output[0].plus(input[6].shl::<2>());
    output[1] = output[1].plus(input[6]);
    output
}

/// Reference `Combine3` (`HalftimeHash24`).
#[inline(always)]
pub(crate) fn combine3<B: Block>(input: &[B; 9]) -> [B; 3] {
    let mut output = [B::zero(); 3];
    output[1] = input[0];
    output[2] = input[0];
    output[1] = output[1].plus(input[1]);
    output[2] = output[2].plus(input[1].shl::<2>());
    output[0] = input[2];
    output[2] = output[2].plus(input[2]);
    output[0] = output[0].plus(input[3].shl::<2>());
    output[2] = output[2].plus(input[3]);
    output[0] = output[0].plus(input[4]);
    output[1] = output[1].plus(input[4]);
    output[0] = output[0].plus(input[5]);
    output[1] = output[1].plus(input[5].shl::<2>());
    // Dot3<2, 1, 2>
    output[0] = output[0].plus(input[6].shl::<1>());
    output[1] = output[1].plus(input[6]);
    output[2] = output[2].plus(input[6].shl::<1>());
    // Dot3<2, 2, 1>
    output[0] = output[0].plus(input[7].shl::<1>());
    output[1] = output[1].plus(input[7].shl::<1>());
    output[2] = output[2].plus(input[7]);
    // Dot3<1, 2, 2>
    output[0] = output[0].plus(input[8]);
    output[1] = output[1].plus(input[8].shl::<1>());
    output[2] = output[2].plus(input[8].shl::<1>());
    output
}

/// Reference `Combine4` (`HalftimeHash32`).
#[inline(always)]
pub(crate) fn combine4<B: Block>(input: &[B; 10]) -> [B; 4] {
    let mut output = [B::zero(); 4];
    output[2] = input[0].shl::<1>();
    output[3] = input[0];
    output[1] = input[1];
    output[3] = output[3].plus(input[1]);
    output[1] = output[1].plus(input[2].shl::<1>());
    output[2] = output[2].plus(input[2]);
    output[0] = input[3];
    output[3] = output[3].plus(input[3]);
    output[0] = output[0].plus(input[4]);
    output[2] = output[2].plus(input[4].shl::<2>());
    output[0] = output[0].plus(input[5].shl::<2>());
    output[1] = output[1].plus(input[5]);
    // Dot4<2, 1, 1, 4>
    output[0] = output[0].plus(input[6].shl::<1>());
    output[1] = output[1].plus(input[6]);
    output[2] = output[2].plus(input[6]);
    output[3] = output[3].plus(input[6].shl::<2>());
    // Dot4<4, 2, 1, 1>
    output[0] = output[0].plus(input[7].shl::<2>());
    output[1] = output[1].plus(input[7].shl::<1>());
    output[2] = output[2].plus(input[7]);
    output[3] = output[3].plus(input[7]);
    // Dot4<1, 4, 1, 2>
    output[0] = output[0].plus(input[8]);
    output[1] = output[1].plus(input[8].shl::<2>());
    output[2] = output[2].plus(input[8]);
    output[3] = output[3].plus(input[8].shl::<1>());
    // Dot4<1, 1, 1, 8>
    output[0] = output[0].plus(input[9]);
    output[1] = output[1].plus(input[9]);
    output[2] = output[2].plus(input[9]);
    output[3] = output[3].plus(input[9].shl::<3>());
    output
}

/// Reference `Combine5` (`HalftimeHash40`).
#[inline(always)]
pub(crate) fn combine5<B: Block>(input: &[B; 9]) -> [B; 5] {
    let mut output = [input[0], input[1], input[2], input[3], input[4]];
    output[0] = output[0].plus(input[5]);
    output[1] = output[1].plus(input[5]);
    output[2] = output[2].plus(input[5]);
    output[3] = output[3].plus(input[5]);
    output[4] = output[4].plus(input[5]);
    // Dot5<1, 2, 3, 4, 5>
    output[0] = output[0].plus(input[6]);
    output[1] = output[1].plus(input[6].shl::<1>());
    output[2] = output[2].plus(input[6].plus(input[6].shl::<1>()));
    output[3] = output[3].plus(input[6].shl::<2>());
    output[4] = output[4].plus(input[6].plus(input[6].shl::<2>()));
    // Dot5<2, 1, 8, 9, 3>
    output[0] = output[0].plus(input[7].shl::<1>());
    output[1] = output[1].plus(input[7]);
    output[2] = output[2].plus(input[7].shl::<3>());
    output[3] = output[3].plus(input[7].plus(input[7].shl::<3>()));
    output[4] = output[4].plus(input[7].plus(input[7].shl::<1>()));
    // Dot5<4, 7, 5, 8, 9>
    output[0] = output[0].plus(input[8].shl::<2>());
    output[1] = output[1].plus(input[8].shl::<3>().minus(input[8]));
    output[2] = output[2].plus(input[8].plus(input[8].shl::<2>()));
    output[3] = output[3].plus(input[8].shl::<3>());
    output[4] = output[4].plus(input[8].plus(input[8].shl::<3>()));
    output
}
