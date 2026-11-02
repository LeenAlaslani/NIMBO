import {test} from 'node:test';
import {strict as assert} from 'node:assert';
// Pure validation mirror of M9 intake constraints; browser/backend gate remains authoritative.
const id = x => /^[a-zA-Z0-9][a-zA-Z0-9._-]+\/[a-zA-Z0-9][a-zA-Z0-9._-]+$/.test(x);
const sha = x => /^[0-9a-f]{40}$/.test(x);
const file = x => /^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,90}\.(csv|jsonl)$/.test(x);
test('supported example is an immutable public model ID',()=>{assert.equal(id('cardiffnlp/twitter-roberta-base-sentiment-latest'),true);assert.equal(sha('b995b77f45ae7ed76649b47d04952ec8be9f3591'),true)});
test('unsafe paths and arbitrary archives are rejected',()=>{for(const name of ['../x.csv','/tmp/data.csv','model.onnx','../../../secret.jsonl'])assert.equal(file(name),false)});
test('revision must be 40 lowercase hex digits',()=>{assert.equal(sha('a'.repeat(40)),true);assert.equal(sha('Z'.repeat(40)),false);assert.equal(sha('a'.repeat(39)),false)});
test('model IDs require owner/model and no traversal',()=>{assert.equal(id('org/model'),true);assert.equal(id('org/../x'),false)});
