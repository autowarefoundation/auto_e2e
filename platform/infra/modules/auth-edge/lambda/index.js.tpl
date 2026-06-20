'use strict';
const { Authenticator } = require('cognito-at-edge');

const authenticator = new Authenticator({
  region: '${region}',
  userPoolId: '${user_pool_id}',
  userPoolAppId: '${client_id}',
  userPoolAppSecret: '${client_secret}',
  userPoolDomain: '${cognito_domain}',
  cookieExpirationDays: 7,
});

exports.handler = async (event) => authenticator.handle(event);
